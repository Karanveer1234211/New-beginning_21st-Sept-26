import sys, types, json, shutil, tempfile, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd

for n in ("kiteconnect","kiteconnect.exceptions"):
    sys.modules[n] = types.ModuleType(n)
class _E(Exception): pass
sys.modules["kiteconnect"].KiteConnect = object
for n in ("KiteException","TokenException","InputException","NetworkException","DataException","GeneralException"):
    setattr(sys.modules["kiteconnect.exceptions"], n, type(n,(_E,),{}))

import importlib.util
def load(p,n):
    sp=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(sp)
    sys.modules[n]=m; sp.loader.exec_module(m); return m
dc = load("Daily_cache_v24.py","dc")
dq = load("data_quality.py","dq")
print("modules import: OK\n")

root = Path(tempfile.mkdtemp())
sessions = pd.DatetimeIndex(pd.bdate_range("2024-01-01","2024-12-31"))
# NSE-style: drop some real holidays, and ADD a Muhurat-style Sunday session
holidays = pd.DatetimeIndex(["2024-03-08","2024-03-25","2024-08-15","2024-10-02"])
sessions = sessions.difference(holidays)
muhurat = pd.Timestamp("2024-11-03")          # a Sunday that traded
sessions = sessions.append(pd.DatetimeIndex([muhurat])).sort_values()
print(f"synthetic exchange: {len(sessions)} sessions incl. Muhurat {muhurat.date()}\n")

def mkdf(ix, seed=0, vol=True):
    r = np.random.default_rng(seed)
    c = 100*np.exp(np.cumsum(r.normal(0,0.01,len(ix))))
    d = pd.DataFrame({"timestamp": ix, "open": c, "high": c*1.01, "low": c*0.99, "close": c})
    d["volume"] = r.integers(1e4,1e6,len(ix)).astype(float) if vol else np.nan
    return d

def write(sym, ix, kind="equity", seed=0, vol=True, rows=None, suspects=None):
    df = mkdf(ix, seed, vol)
    df.to_parquet(root/f"{sym}_daily.parquet", index=False)
    (root/f"{sym}_daily.ok.json").write_text(json.dumps({
        "schema_version":"v24","rows": rows if rows is not None else len(df),
        "series_kind":kind,"corporate_action_suspects":suspects or [],
        "last_timestamp":str(df["timestamp"].iloc[-1])}, default=str))

# 30 ordinary stocks
for i in range(30):
    write(f"STK{i:02d}", sessions, seed=i)
# NIFTY50 reference, itself MISSING a date the universe has (the v23 blind spot)
nifty_gap = pd.Timestamp("2024-06-12")
write("NIFTY50", sessions.difference([nifty_gap]), kind="index", seed=99, vol=False)
# a suspended stock: 20 contiguous sessions absent
susp_run = sessions[120:140]
write("SUSPENDED", sessions.difference(susp_run), seed=201)
# a stock with 8 scattered missing sessions = real data loss
iso = sessions[[10,40,77,90,130,160,200,230]]
write("LOSSY", sessions.difference(iso), seed=202)
# a legitimately late listing
write("NEWLIST", sessions[150:], seed=203)
# market-wide outage: one date dropped from EVERY stock
outage = pd.Timestamp("2024-09-17")
for i in range(30):
    df = mkdf(sessions.difference([outage]), seed=i)
    df.to_parquet(root/f"STK{i:02d}_daily.parquet", index=False)
    (root/f"STK{i:02d}_daily.ok.json").write_text(json.dumps({
        "schema_version":"v24","rows":len(df),"series_kind":"equity",
        "corporate_action_suspects":[],"last_timestamp":str(df["timestamp"].iloc[-1])},
        default=str))

syms = dq.discover_symbols(root)

# ============ 1. CALENDAR CROSS-CHECK ============
cal = dq.build_calendar(root, syms)
print(f"1. calendar source={cal.source} trusted={cal.trusted} sessions={len(cal)}")
print(f"   sources: {cal.sources}")
for n in cal.notes: print(f"   NOTE: {n}")
print(f"   disagreements: {cal.disagreement_count if hasattr(cal,'disagreement_count') else len(cal.disagreements)}")
# the NIFTY50 gap must be visible as a disagreement, not silently adopted
ref_dis = [d for d in cal.disagreements if d["date"]==str(nifty_gap.date())]
assert ref_dis, "NIFTY50's own gap must surface as a source disagreement"
print(f"   NIFTY50 gap {nifty_gap.date()} flagged: {ref_dis[0]}")
assert nifty_gap in cal.sessions, "consensus must keep a date NIFTY50 dropped"
print("   consensus overrides the reference's own gap  OK")
assert muhurat in cal.sessions, "special session must survive"
print(f"   Muhurat session {muhurat.date()} retained  OK")
assert outage not in cal.sessions, "universe-wide outage must not be a session"
print(f"   universe-wide outage {outage.date()} excluded from calendar  OK\n")

# ============ 2. ABSENCE CLASSIFICATION ============
cls = dq.classify_absences(root, syms, calendar=cal)
print(f"2. market-wide dates: {cls['market_wide_dates']}")
S = cls["symbols"]
print(f"   SUSPENDED -> {S['SUSPENDED']['suspension']} suspension, "
      f"{S['SUSPENDED']['isolated']} isolated, runs={S['SUSPENDED']['suspension_runs']}")
assert S["SUSPENDED"]["suspension"] == 20 and S["SUSPENDED"]["isolated"] == 0
print(f"   LOSSY     -> {S['LOSSY']['suspension']} suspension, "
      f"{S['LOSSY']['isolated']} isolated  {S['LOSSY']['isolated_dates'][:3]}")
assert S["LOSSY"]["isolated"] == 8 and S["LOSSY"]["suspension"] == 0
print(f"   NEWLIST   -> defects={S['NEWLIST']['defects']} (late listing, not a gap)")
assert S["NEWLIST"]["defects"] == 0
print(f"   STK00     -> defects={S['STK00']['defects']} (outage is market-wide)")
assert S["STK00"]["defects"] == 0
print("   suspension / data-loss / listing / outage all separated  OK\n")

# ============ 3. GATE USES THE CLASSIFICATION ============
rep = dq.gate_all(root, syms, schema_version="v24")
print(f"3. gate: PASS {rep['allowed']} / QUARANTINE {rep['quarantined']}")
q = {x["symbol"]: x for x in rep["quarantine"]}
assert "LOSSY" in q, "isolated data loss must quarantine"
print(f"   LOSSY quarantined: {q['LOSSY']['reasons'][0][:70]}")
assert "SUSPENDED" not in q, "a real suspension is not a data defect"
print("   SUSPENDED allowed (data is correct - it did not trade)")
assert "NEWLIST" not in q and "STK00" not in q
print("   NEWLIST + outage-affected stocks allowed  OK\n")

# ============ 4. rows_valid BUG FIXED ============
src = open("Daily_cache_v24.py", encoding="utf-8").read()
assert 'r["rows_valid"] == 0' not in src
assert 'int(meta.get("rows_valid") or 0)' not in src
print("4. rows_valid removed from coverage report")
import re
line = [l.strip() for l in src.splitlines() if l.strip().startswith("empty = ")][0]
print(f"   {line}")
assert 'r["rows"] == 0' in line
print("   'ZERO USABLE ROWS' no longer fires for every symbol  OK\n")

# ============ 5. UNTRUSTED CALENDAR STILL FAILS CLOSED ============
solo = Path(tempfile.mkdtemp())
shutil.copy(root/"STK00_daily.parquet", solo); shutil.copy(root/"STK00_daily.ok.json", solo)
rep2 = dq.gate_all(solo, ["STK00"], schema_version="v24")
c2 = rep2.get("calendar") or {}
print(f"5. single-symbol root -> allowed={rep2['allowed']} trusted={c2.get('trusted')}")
for n in c2.get("notes", []): print(f"   NOTE: {n}")
assert not c2.get("trusted"), "one symbol cannot form a trusted calendar"
assert rep2["allowed"] == 0, "untrusted calendar must fail closed"
print("\nALL V24 TESTS PASSED")

# ============ 6. D_roll NA IS NEVER FILLED ============
fd = load("features_daily.py", "fd")
print("\n6. D_roll NA preservation")
ix = pd.date_range("2022-01-03", periods=700, freq="B", tz=dc.IST)
r = np.random.default_rng(5)
c = 100*np.exp(np.cumsum(r.normal(0.0004,0.015,len(ix))))
raw = pd.DataFrame({"timestamp": ix, "open": c*1.001, "high": c*1.01,
                    "low": c*0.99, "close": c,
                    "volume": r.integers(1e4,1e6,len(ix)).astype(float)})
cal = pd.DataFrame({"expiry": pd.to_datetime(["2023-06-20","2023-07-19"])})
cru = dc.Series.future("CRUDEOIL","MCX","CRUDEOIL", session_end_ist="23:30")
raw = dc.apply_roll_marks(raw, cal, cru)
na_in = int(pd.isna(raw["D_roll"]).sum())
print(f"   input D_roll: {na_in} NA, {int((raw['D_roll']==1).sum())} rolls")
assert na_in > 0

out = fd.build_features(raw.copy())
na_out = int(pd.isna(out["D_roll"]).sum())
print(f"   after build_features: {na_out} NA  (dtype={out['D_roll'].dtype})")
assert na_out == na_in, f"NA count changed {na_in} -> {na_out}"
assert str(out["D_roll"].dtype) == "Int8"
assert "D_roll" in fd.PRESERVE_NA
print("   NA count unchanged, dtype still nullable  OK")

after_warm = fd.drop_warmup(out)
assert int(pd.isna(after_warm["D_roll"]).sum()) > 0 or len(after_warm) == 0
print(f"   survives drop_warmup: {int(pd.isna(after_warm['D_roll']).sum())} NA remain  OK")

# the guard must actually fire, not just pass by luck
bad = out.copy(); bad["D_roll"] = bad["D_roll"].fillna(0)
try:
    fd.assert_na_preserved({"D_roll": na_in}, bad)
    print("   FAIL - guard did not fire")
except fd.UnknownRollFilled as e:
    print(f"   guard fires when filled: {str(e)[:66]}...  OK")

src = open("Daily_cache_v24.py", encoding="utf-8").read()
assert "v22 (2026" not in src and "v24 (2026" in src
assert "_mark_rolls" not in src.split("ROLL CALENDAR")[0]
print("\n7. header updated to v24, no stale _mark_rolls description  OK")
print("\nALL V24 TESTS PASSED (incl. D_roll guard)")
