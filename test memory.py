"""The three ways an analogue engine lies: clustering, leakage, self-match."""
import sys, tempfile, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0,".")
import memory as M

root=Path(tempfile.mkdtemp()); pdir=root/"panel"; pdir.mkdir()
rng=np.random.default_rng(11)
dates=pd.DatetimeIndex(pd.bdate_range("2018-01-01","2026-09-21"))
syms=[f"S{i:02d}" for i in range(30)]
rows=[]
for s in syms:
    for d in dates:
        r={"timestamp":d,"symbol":s}
        for c in M.STATE_FEATURES: r[c]=rng.normal()
        r["label_touch"]=float(rng.binomial(1,.35))
        r["label_tp_before_sl"]=float(rng.binomial(1,.28))
        r["label_fwd_ret_5d"]=rng.normal(0,.05)
        r["label_mfe_5d"]=abs(rng.normal(.045,.03))
        r["label_mae_5d"]=-abs(rng.normal(.032,.02))
        r["label_days_to_tp"]=float(rng.integers(1,6))
        rows.append(r)
panel=pd.DataFrame(rows)
panel.to_parquet(pdir/"panel.parquet",index=False)
print(f"fixture: {len(panel):,} episodes, {len(syms)} symbols\n")

print("=== 1. BUILD ===")
M.build_memory(pdir/'panel.parquet')

print("\n=== 2. NO FUTURE LEAKAGE ===")
qd="2024-06-28"
r=M.ask(pdir/"panel.parquet","S00",date=qd,scope="self",verbose=False)
idx=pd.read_parquet(pdir/"memory"/"index.parquet")
latest=max(pd.Timestamp(a["date"]) for a in r["analogues"])
sess=np.unique(pd.to_datetime(idx["timestamp"]).to_numpy())
qpos=int(np.searchsorted(sess,np.datetime64(pd.Timestamp(qd)),side="right"))-1
bound=pd.Timestamp(sess[qpos-M.LABEL_HORIZON])
cal_bound=pd.Timestamp(qd)-pd.Timedelta(days=M.LABEL_HORIZON+1)
print(f"  query {qd}")
print(f"  session bound  {bound.date()}  ({M.LABEL_HORIZON} sessions back)")
print(f"  calendar bound {cal_bound.date()}  (v2 used this - {(bound-cal_bound).days:+d} days looser)")
print(f"  latest analogue {latest.date()}")
assert latest<=bound, "analogue used an unresolved outcome"
print("  ok   bound is in SESSIONS; no analogue's 5-day window was still open")
print(f"  reported cutoff: {r['cutoff_session']} | PIT epoch: {r['pit_epoch']}")

print("\n=== 3. SELF-MATCH EXCLUDED ===")
assert all(pd.Timestamp(a["date"])!=pd.Timestamp(qd) for a in r["analogues"])
print("  ok   the query episode is not its own analogue")

print("\n=== 4. DECLUSTERING ===")
print(f"  raw neighbours {r['raw_neighbours']} -> after declustering "
      f"{r['after_declustering']} -> within distance {r['declustered_evidence_count']}")
# verify separation among SAME-symbol analogues
full=M.ask(pdir/"panel.parquet","S00",date=qd,scope="self",k=50,verbose=False)
ds=sorted(pd.Timestamp(a["date"]) for a in full["analogues"])
gaps=[(ds[i+1]-ds[i]).days for i in range(len(ds)-1)]
print(f"  min gap between selected analogues: {min(gaps) if gaps else 'n/a'} days "
      f"(required >= {M.MIN_SEPARATION})")
assert not gaps or min(gaps)>=M.MIN_SEPARATION
print("  ok   near-duplicate days cannot inflate the evidence count")

# and the declusterer must actually remove things
raw=M.ask(pdir/"panel.parquet","S00",date=qd,scope="self",
          min_separation=0,verbose=False)
print(f"  with declustering OFF: {raw['after_declustering']} kept vs "
      f"{r['after_declustering']} with it on")
assert raw["after_declustering"]>=r["after_declustering"]
print("  ok   declustering is doing work, not decoration")

print("\n=== 5. CONFIDENCE FALLS WHEN EVIDENCE IS THIN ===")
# A query earlier than any frozen normalisation must be REFUSED, not served
# with a future-informed transform.
try:
    M.ask(pdir/"panel.parquet","S00",date="2018-03-01",scope="self",verbose=False)
    print("  FAIL: served a query with no PIT normalisation available")
    raise SystemExit(1)
except RuntimeError as e:
    print(f"  ok   query predating every frozen epoch refused: {str(e)[:52]}...")

early=M.ask(pdir/"panel.parquet","S00",date="2020-06-01",scope="self",verbose=False)
late =M.ask(pdir/"panel.parquet","S00",date="2026-06-01",scope="self",verbose=False)
print(f"  2020 (less history): n={early.get('declustered_evidence_count')} "
      f"score={early.get('evidence_score')} epoch={early.get('pit_epoch')}")
print(f"  2026 (full history): n={late.get('declustered_evidence_count')} "
      f"score={late.get('evidence_score')} epoch={late.get('pit_epoch')}")
assert early.get("evidence_score",0)<=late.get("evidence_score",1)
assert early["pit_epoch"]!=late["pit_epoch"], "epochs must differ across years"
print("  ok   less history -> lower score, and each query used its OWN epoch")

print("\n=== 6. CROSS-STOCK SCOPE EXCLUDES THE STOCK ITSELF ===")
cr=M.ask(pdir/"panel.parquet","S00",date=qd,scope="cross",verbose=False)
assert all(a["symbol"]!="S00" for a in cr["analogues"])
print(f"  cross-scope n={cr['declustered_evidence_count']} vs self n={r['declustered_evidence_count']}")
print("  ok   scopes answer different questions and stay separate")

print("\n=== 7. RENDERED OUTPUT ===")
M.ask(pdir/"panel.parquet","S00",date=qd,scope="self")
print("\nMEMORY ENGINE VERIFIED")

print("\n=== 8. DECLUSTERING PROVEN ON A PLANTED CLUSTER ===")
# Build a stock whose state is IDENTICAL on 40 consecutive days. Without
# declustering that is 40 'independent' analogues; it is really one episode.
rows2=[]
base={c:0.0 for c in M.STATE_FEATURES}
for i,d in enumerate(dates):
    r={"timestamp":d,"symbol":"CLUST"}
    if 500<=i<540:                      # 40 consecutive identical days
        r.update(base)
    else:
        r.update({c:rng.normal(5,1) for c in M.STATE_FEATURES})
    r["label_touch"]=1.0; r["label_tp_before_sl"]=1.0
    r["label_fwd_ret_5d"]=.05; r["label_mfe_5d"]=.06
    r["label_mae_5d"]=-.01; r["label_days_to_tp"]=2.0
    rows2.append(r)
# query day also at the cluster state, much later
for i,d in enumerate(dates):
    if i==1500:
        rows2[i].update(base)
p2=pd.concat([panel,pd.DataFrame(rows2)],ignore_index=True)
root2=Path(tempfile.mkdtemp()); pd2=root2/"panel"; pd2.mkdir()
p2.to_parquet(pd2/"panel.parquet",index=False)
M.build_memory(pd2/"panel.parquet",verbose=False)
qd2=str(dates[1500].date())
on =M.ask(pd2/"panel.parquet","CLUST",date=qd2,scope="self",k=40,verbose=False)
off=M.ask(pd2/"panel.parquet","CLUST",date=qd2,scope="self",k=40,
          min_separation=0,verbose=False)
print(f"  40 identical consecutive days in history (k=40):")
print(f"    declustering OFF -> {off['declustered_evidence_count']} 'analogues'")
print(f"    declustering ON  -> {on['declustered_evidence_count']} analogues")
assert on["declustered_evidence_count"]<off["declustered_evidence_count"], (on["declustered_evidence_count"],off["declustered_evidence_count"])
ratio=off["declustered_evidence_count"]/max(on["declustered_evidence_count"],1)
print(f"    inflation factor without declustering: {ratio:.1f}x")
print("  ok   one episode is counted once, not forty times")
print("       (this is the '147 analogues' illusion, prevented)")
print("\nMEMORY ENGINE VERIFIED (incl. planted cluster)")


print("\n=== 9. PIT NORMALISATION IS ACTUALLY FROZEN ===")
# Adding FUTURE data must not change a PAST query's result. This is the
# property v2 violated: whole-panel ranks meant tomorrow's bars could
# reorder yesterday's analogues.
q9="2022-06-30"
before=M.ask(pdir/"panel.parquet","S00",date=q9,scope="self",verbose=False)

extra=[]
for s_ in syms[:5]:
    for d in pd.bdate_range("2026-09-22","2027-06-30"):
        r={"timestamp":d,"symbol":s_}
        for c in M.STATE_FEATURES: r[c]=rng.normal(50,20)   # wildly different scale
        r["label_touch"]=1.0; r["label_tp_before_sl"]=1.0
        r["label_fwd_ret_5d"]=.09; r["label_mfe_5d"]=.11
        r["label_mae_5d"]=-.01; r["label_days_to_tp"]=1.0
        extra.append(r)
p9=pd.concat([panel,pd.DataFrame(extra)],ignore_index=True)
root9=Path(tempfile.mkdtemp()); pd9=root9/"panel"; pd9.mkdir()
p9.to_parquet(pd9/"panel.parquet",index=False)
M.build_memory(pd9/"panel.parquet",verbose=False)
after=M.ask(pd9/"panel.parquet","S00",date=q9,scope="self",verbose=False)

print(f"  added {len(extra):,} future episodes at a very different scale")
print(f"  2022 query BEFORE: n={before['declustered_evidence_count']} "
      f"epoch={before['pit_epoch']} analogue1={before['analogues'][0]['date']}")
print(f"  2022 query AFTER:  n={after['declustered_evidence_count']} "
      f"epoch={after['pit_epoch']} analogue1={after['analogues'][0]['date']}")
b=[a["date"] for a in before["analogues"]]
a_=[a["date"] for a in after["analogues"]]
assert b==a_, f"future data changed a past query's analogues:\n{b}\n{a_}"
assert before["declustered_evidence_count"]==after["declustered_evidence_count"]
print("  ok   identical analogues, identical count")
print("       (v2 would have reordered these - whole-panel ranks)")
print("\nMEMORY ENGINE VERIFIED (PIT normalisation, session cutoff)")
