import sys, types, json, tempfile, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd
for n in ("kiteconnect","kiteconnect.exceptions"): sys.modules[n]=types.ModuleType(n)
sys.modules["kiteconnect"].KiteConnect=object
for n in ("KiteException","TokenException","InputException","NetworkException","DataException","GeneralException"):
    setattr(sys.modules["kiteconnect.exceptions"],n,type(n,(Exception,),{}))
sys.path.insert(0,".")
import importlib
pb = importlib.import_module("panel_build")
fx = importlib.import_module("features_daily")

root=Path(tempfile.mkdtemp()); pdir=Path(tempfile.mkdtemp())
ses=pd.DatetimeIndex(pd.bdate_range("2021-01-01","2024-12-31"))
IST=fx.IST
def w(sym,ix,kind="equity",seed=0,vol=True,exch=None):
    r=np.random.default_rng(seed)
    c=100*np.exp(np.cumsum(r.standard_t(4,len(ix))*0.013))
    o=c*(1+r.normal(0,.004,len(ix)))
    d=pd.DataFrame({"timestamp":ix.tz_localize(IST),"open":o,
                    "high":np.maximum(c,o)*1.008,"low":np.minimum(c,o)*0.992,"close":c,
                    "volume":r.lognormal(13.6,.85,len(ix)) if vol else np.nan})
    d.to_parquet(root/f"{sym}_daily.parquet",index=False)
    m={"schema_version":"v26","rows":len(d),"series_kind":kind,
       "corporate_action_suspects":[],"last_timestamp":str(d["timestamp"].iloc[-1])}
    if exch: m["exchange"]=exch
    (root/f"{sym}_daily.ok.json").write_text(json.dumps(m,default=str))

N=30
for i in range(N): w(f"STK{i:02d}",ses,seed=i)
w("NIFTY50",ses,kind="index",seed=900,vol=False)
print(f"fixture: {N} equities + index, {len(ses)} sessions\n")

print("=== FULL BUILD ===")
p=pb.build_panel(root,pdir,full=True)
panel=pd.read_parquet(p)
meta=json.loads((pdir/"panel_meta.json").read_text())
print()

print("=== CHECKS ===")
# 1. labels: last horizon sessions unresolved, NOT zero
last=sorted(panel["timestamp"].unique())[-pb.LABEL_HORIZON:]
tail=panel[panel["timestamp"].isin(last)]
assert tail["label_touch"].isna().all(), "recent labels must be UNKNOWN"
print(f"  ok   last {pb.LABEL_HORIZON} sessions unlabelled ({len(tail)} rows), not zero-filled")

# 2. label is genuinely forward and correct on a spot check
s0=panel[panel.symbol=="STK00"].sort_values("timestamp").reset_index(drop=True)
raw=pd.read_parquet(root/"STK00_daily.parquet")
raw["timestamp"]=pd.to_datetime(raw["timestamp"]).dt.tz_localize(None)
j=len(s0)//2; d0=s0.loc[j,"timestamp"]
ri=raw.index[raw.timestamp==d0][0]
fwd=raw.loc[ri+1:ri+5,"high"].max()/raw.loc[ri,"close"]-1
assert abs(fwd-s0.loc[j,"label_mfe_5d"])<1e-12
assert int(s0.loc[j,"label_touch"])==int(fwd>=0.05)
print(f"  ok   label_touch matches hand-computed 5-day MFE ({fwd:+.3%})")

# 3. no label leaks into X
X=pb.panel_feature_columns(panel)
assert not any(c.startswith("label_") for c in X)
assert "ret_5d_close_pct" not in X
print(f"  ok   X has {len(X)} cols, zero label/forward columns")

# 4. cross-sectional ranks are within-date
g=panel.groupby("timestamp")["X_rank_D_rsi14"]
mx=g.max().dropna(); mn=g.min().dropna()
assert (mx<=1.0+1e-9).all() and (mn>=0).all()
print(f"  ok   cross-sectional ranks in [0,1] within each date")

# 5. point-in-time universe: a late lister must not appear before listing
w("LATE",ses[400:],seed=777)
p2=pb.build_panel(root,pdir,full=True,verbose=False)
pn2=pd.read_parquet(p2)
first_late=pn2[pn2.symbol=="LATE"]["timestamp"].min()
assert first_late>=ses[400]
others=pn2[(pn2.timestamp<ses[400])]["symbol"].unique()
assert "LATE" not in others
print(f"  ok   late lister absent before {first_late.date()} (point-in-time universe)")

# 6. incremental == full: EXACT equivalence, not just numeric maxdiff
pn_full=pd.read_parquet(p2).copy()
p3=pb.build_panel(root,pdir,full=False,rebuild_days=15,verbose=False)
pn_inc=pd.read_parquet(p3)

a=pn_full.sort_values(["timestamp","symbol"]).reset_index(drop=True)
b=pn_inc.sort_values(["timestamp","symbol"]).reset_index(drop=True)

assert list(a.columns)==list(b.columns), (
    f"column sets differ: only-full={set(a.columns)-set(b.columns)}, "
    f"only-inc={set(b.columns)-set(a.columns)}")
print(f"  ok   identical column list ({len(a.columns)})")

dt_diff={c:(str(a[c].dtype),str(b[c].dtype)) for c in a.columns if a[c].dtype!=b[c].dtype}
assert not dt_diff, f"dtype drift: {dt_diff}"
print(f"  ok   identical dtypes across all {len(a.columns)} columns")

assert len(a)==len(b), f"row count {len(a)} vs {len(b)}"
assert a[["timestamp","symbol"]].equals(b[["timestamp","symbol"]]), "key order differs"
print(f"  ok   identical row keys ({len(a):,} rows)")

nan_mismatch=[c for c in a.columns if not a[c].isna().equals(b[c].isna())]
assert not nan_mismatch, f"NaN masks differ: {nan_mismatch[:6]}"
print(f"  ok   identical NaN masks (labels included)")

bad=[]
for c in a.columns:
    if pd.api.types.is_numeric_dtype(a[c]):
        av=a[c].to_numpy(dtype="float64",na_value=np.nan)
        bv=b[c].to_numpy(dtype="float64",na_value=np.nan)
        if not np.array_equal(av,bv,equal_nan=True): bad.append(c)
    else:
        if not a[c].equals(b[c]): bad.append(c)
assert not bad, f"value mismatch in {bad[:6]}"
print(f"  ok   bit-exact values in every column (numeric and non-numeric)")

# 7. embargo is enforced
try:
    pb.walk_forward_splits(pn_inc,n_splits=3,embargo=2); print("  FAIL embargo not enforced")
except ValueError as e: print(f"  ok   embargo < horizon rejected: {str(e)[:52]}")
folds=pb.walk_forward_splits(pn_inc,n_splits=3)
for f in folds:
    assert pd.Timestamp(f["train_end"])<pd.Timestamp(f["test_start"])
print(f"  ok   {len(folds)} walk-forward folds, train_end < test_start with embargo")
print(f"       e.g. fold1 train->{folds[0]['train_end']} test {folds[0]['test_start']}..{folds[0]['test_end']}")

# 8. universe change forces a full rebuild, and quarantined history is dropped
import shutil
(root/"STK07_daily.parquet").unlink()
(root/"STK07_daily.ok.json").unlink()
p4=pb.build_panel(root,pdir,full=False,rebuild_days=15,verbose=True)
pn4=pd.read_parquet(p4)
assert "STK07" not in set(pn4["symbol"]), "removed symbol must vanish from ALL history"
m4=json.loads((pdir/"panel_meta.json").read_text())
assert m4["mode"]=="full", "universe change must force a full rebuild"
print(f"  ok   universe change -> full rebuild; STK07 gone from all "
      f"{pn4['timestamp'].nunique()} sessions")

print(f"\n  panel: {meta['rows']:,} rows, {meta['columns']} cols, "
      f"base rate {meta['label_positive_rate']:.2%}")
print("\nALL PANEL TESTS PASSED")
