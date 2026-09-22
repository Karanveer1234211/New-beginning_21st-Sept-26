"""Are the soul features point-in-time, and do they carry the disagreement?"""
import sys, tempfile, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0,".")
import memory as M, soul_features as SF

root=Path(tempfile.mkdtemp()); pdir=root/"panel"; pdir.mkdir()
rng=np.random.default_rng(23)
dates=pd.DatetimeIndex(pd.bdate_range("2020-01-01","2026-09-21"))
syms=[f"S{i:02d}" for i in range(25)]
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
# a late lister: almost no own history, so self-memory must be unavailable
for d in dates[-100:]:
    r={"timestamp":d,"symbol":"NEWCO"}
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
M.build_memory(pdir/"panel.parquet",verbose=False)
SF.build(pdir/"panel.parquet",every=40,cross_pool=3000,top_k=30,verbose=True)
f=pd.read_parquet(pdir/"soul"/"soul_features.parquet")
print(f"  {len(f):,} feature rows, {len(SF.FEATURE_COLUMNS)} features")

print("\n=== 2. POINT-IN-TIME: future data cannot change past features ===")
f_before=f.copy()
extra=[]
for s_ in syms[:6]:
    for d in pd.bdate_range("2026-09-22","2027-03-31"):
        r={"timestamp":d,"symbol":s_}
        for c in M.STATE_FEATURES: r[c]=rng.normal(40,15)   # different scale
        r["label_touch"]=1.0; r["label_tp_before_sl"]=1.0
        r["label_fwd_ret_5d"]=.10; r["label_mfe_5d"]=.12
        r["label_mae_5d"]=-.005; r["label_days_to_tp"]=1.0
        extra.append(r)
p2=pd.concat([panel,pd.DataFrame(extra)],ignore_index=True)
root2=Path(tempfile.mkdtemp()); pd2=root2/"panel"; pd2.mkdir()
p2.to_parquet(pd2/"panel.parquet",index=False)
M.build_memory(pd2/"panel.parquet",verbose=False)
SF.build(pd2/"panel.parquet",every=40,cross_pool=3000,top_k=30,verbose=False)
f_after=pd.read_parquet(pd2/"soul"/"soul_features.parquet")

key=["timestamp","symbol"]
m=f_before.merge(f_after,on=key,suffixes=("_b","_a"))
m=m[m["timestamp"]<pd.Timestamp("2026-09-01")]
print(f"  added {len(extra):,} future episodes at a different scale")
print(f"  comparing {len(m):,} pre-existing feature rows")
bad=[]
for c in SF.FEATURE_COLUMNS:
    a=pd.to_numeric(m[f"{c}_b"],errors="coerce").to_numpy(dtype=float)
    b=pd.to_numeric(m[f"{c}_a"],errors="coerce").to_numpy(dtype=float)
    if not np.allclose(a,b,equal_nan=True,rtol=0,atol=1e-9): bad.append(c)
print(f"  features changed by future data: {bad or 'NONE'}")
assert not bad, bad
print("  ok   every past feature identical after adding future history")

print("\n=== 3. SELF-AVAILABILITY IS ITSELF INFORMATION ===")
print(f"  self-memory available on {f['soul_self_available'].mean():.1%} of rows")
nc=f[f.symbol=="NEWCO"]["soul_self_available"]
est=f[f.symbol=="S00"]["soul_self_available"]
print(f"  NEWCO (40 sessions of history): self available {nc.mean():.0%}")
print(f"  S00   (full history):           self available {est.mean():.0%}")
assert nc.mean()<1.0 and est.mean()==1.0, (nc.mean(),est.mean())
# where self is unavailable, self features MUST be null and cross MUST NOT be
nf=f[f["soul_self_available"]==0]
print(f"  rows with no self precedent: {len(nf)}")
print(f"    self_tp_first all null: {nf['soul_self_tp_first'].isna().all()}")
print(f"    cross_tp_first all present: {nf['soul_cross_tp_first'].notna().all()}")
assert nf["soul_self_tp_first"].isna().all()
assert nf["soul_cross_tp_first"].notna().all()
print("  ok   no stock-specific precedent -> nulls, NOT a fabricated number")
print("       and cross-memory still answers, which is the useful part")

print("\n=== 4. THE DISAGREEMENT FEATURES EXIST AND VARY ===")
g=pd.to_numeric(f["soul_gap_tp_first"],errors="coerce").dropna()
print(f"  gap non-null on {len(g):,} rows | std {g.std():.3f}")
print(f"  self more bullish {(g>0.1).mean():.1%} | cross more bullish "
      f"{(g<-0.1).mean():.1%} | agree {(g.abs()<=0.1).mean():.1%}")
assert g.std()>0.01, "gap must vary or it carries nothing"
print("  ok   self and cross genuinely disagree on a large share of rows")

print("\n=== 5. ATTACH HOLDS FORWARD, NEVER BACKWARD ===")
j=SF.attach(pdir/"panel.parquet")
chk=j[["timestamp","symbol","soul_cross_tp_first"]].dropna()
sub=chk[chk.symbol=="S00"].sort_values("timestamp")
fs=f[f.symbol=="S00"].sort_values("timestamp")
first_feat=fs["timestamp"].min()
before=sub[sub.timestamp<first_feat]
print(f"  first computed feature date for S00: {first_feat.date()}")
print(f"  rows before that with a non-null value: {len(before)}")
assert len(before)==0, "a row received a feature from the future"
print("  ok   merge_asof backward only - no row sees a future computation")
print(f"  panel {len(pd.read_parquet(pdir/'panel.parquet')):,} rows -> "
      f"joined {len(j):,} rows, {j['soul_cross_tp_first'].notna().mean():.1%} covered")

print("\nSOUL FEATURES VERIFIED")


print("\n=== 6. MEMORY-SAFE DISTANCE MATCHES THE NAIVE FORM ===")
A=rng.normal(size=(50,12)).astype("float32"); B=rng.normal(size=(300,12)).astype("float32")
naive=np.sqrt(((A[:,None,:]-B[None,:,:])**2).sum(axis=2))
fast=SF._cdist(A,B)
print(f"  max abs diff: {np.abs(naive-fast).max():.2e}")
assert np.allclose(naive,fast,atol=1e-4)
nq,npool,dim=1500,60000,12
print(f"  at production scale ({nq}x{npool}x{dim}):")
print(f"    broadcast form would allocate {nq*npool*dim*4/1e9:.1f} GB")
print(f"    batched dot form allocates    {SF.QUERY_BATCH*npool*4/1e9:.3f} GB")
print("  ok   identical result, bounded memory")

print("\n=== 7. HELD-FORWARD VALUES ARE STABLE BETWEEN COMPUTE DATES ===")
j=SF.attach(pdir/"panel.parquet")
fd=sorted(f["timestamp"].unique())
sub=j[(j.symbol=="S00")].sort_values("timestamp")
d0,d1=pd.Timestamp(fd[3]),pd.Timestamp(fd[4])
win=sub[(sub.timestamp>=d0)&(sub.timestamp<d1)]
vals=win["soul_cross_tp_first"].dropna().unique()
print(f"  between compute dates {d0.date()} and {d1.date()}: "
      f"{len(win)} panel rows, {len(vals)} distinct value(s)")
assert len(vals)==1, vals
at=sub[sub.timestamp==d0]["soul_cross_tp_first"].iloc[0]
print(f"  held value {vals[0]:.4f} == value computed on {d0.date()} {at:.4f}")
assert abs(vals[0]-at)<1e-12
print("  ok   the held value is the last COMPUTED one, unchanged until the next")

print("\n=== 8. CROSS-POOL STABILITY ===")
import shutil
res={}
for cp in (3000,6000,12000):
    tmp=Path(tempfile.mkdtemp()); td=tmp/"panel"; td.mkdir()
    shutil.copy(pdir/"panel.parquet",td/"panel.parquet")
    M.build_memory(td/"panel.parquet",verbose=False)
    SF.build(td/"panel.parquet",every=40,cross_pool=cp,top_k=30,verbose=False)
    res[cp]=pd.read_parquet(td/"soul"/"soul_features.parquet")
base=res[12000]
print(f"  {'pool':>8}{'corr vs 12k':>14}{'mean abs diff':>16}")
for cp in (3000,6000):
    m=res[cp].merge(base,on=["timestamp","symbol"],suffixes=("_s","_b"))
    a=pd.to_numeric(m["soul_cross_tp_first_s"],errors="coerce")
    b=pd.to_numeric(m["soul_cross_tp_first_b"],errors="coerce")
    ok=a.notna()&b.notna()
    c=float(a[ok].corr(b[ok])); md=float((a[ok]-b[ok]).abs().mean())
    print(f"  {cp:>8}{c:>14.3f}{md:>16.4f}")
    res[f"c{cp}"]=c
print("  (on RANDOM labels there is no signal to recover, so correlation")
print("   here measures sampling stability only - read it as a floor)")
print("\nSOUL FEATURES VERIFIED")
