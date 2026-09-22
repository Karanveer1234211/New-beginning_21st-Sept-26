"""The audit's job is to find real signal and reject planted noise."""
import sys, tempfile, types, json
from pathlib import Path
import numpy as np, pandas as pd
for n in ("kiteconnect","kiteconnect.exceptions"): sys.modules[n]=types.ModuleType(n)
sys.modules["kiteconnect"].KiteConnect=object
for n in ("KiteException","TokenException","InputException","NetworkException","DataException","GeneralException"):
    setattr(sys.modules["kiteconnect.exceptions"],n,type(n,(Exception,),{}))
sys.path.insert(0,".")
import feature_audit as FA

rng=np.random.default_rng(31)
dates=pd.DatetimeIndex(pd.bdate_range("2019-01-01","2026-09-21"))
syms=[f"S{i:02d}" for i in range(60)]
rows=[]
for d in dates:
    for s in syms:
        rows.append({"timestamp":d,"symbol":s})
p=pd.DataFrame(rows)
n=len(p)
# ---- planted structure ----
p["D_rsi14"]=rng.uniform(20,80,n)
p["D_atr_pct"]=np.abs(rng.normal(3,1,n))
p["D_ema20_angle_deg"]=rng.normal(0,10,n)
p["D_dvol_z20"]=rng.normal(0,1,n)
p["D_pos_in_52w_range"]=rng.uniform(0,1,n)
p["SIGNAL_real"]=rng.normal(0,1,n)             # genuinely predictive
p["NOISE_pure"]=rng.normal(0,1,n)              # nothing
p["NOISE_dup"]=p["NOISE_pure"]*2.0+1e-9        # exact duplicate (affine)
p["CONST_dead"]=1.0                            # constant
p["SPARSE_bad"]=np.where(rng.random(n)<0.2,rng.normal(size=n),np.nan)
p["REDUNDANT_copy"]=p["SIGNAL_real"]*0.98+rng.normal(0,0.05,n)
# target depends ONLY on SIGNAL_real
lin=1.6*p["SIGNAL_real"]
p["label_tp_before_sl"]=(rng.random(n)<1/(1+np.exp(-lin))).astype(float)
for c in ("close","D_ema20","D_ema50","D_atr14","D_volume","D_vol_sma20"):
    p[c]=np.abs(rng.normal(100,10,n))
root=Path(tempfile.mkdtemp()); pdir=root/"panel"; pdir.mkdir()
p.to_parquet(pdir/"panel.parquet",index=False)
print(f"fixture: {len(p):,} rows, 1 real signal + planted junk\n")

FA.run(pdir/"panel.parquet",mode="marginal",top_candidates=10,n_splits=3,
       max_rows=200_000,verbose=True)
a=pd.read_parquet(pdir/"audit"/"feature_audit.parquet").set_index("feature")

print("\n=== VERDICTS ON PLANTED FEATURES ===")
for f in ("SIGNAL_real","NOISE_pure","NOISE_dup","CONST_dead","SPARSE_bad","REDUNDANT_copy"):
    if f in a.index:
        r=a.loc[f]
        print(f"  {f:<16}{r['status']:<24}IC={r['ic']:+.4f} "
              f"dAUC={r['delta_mean'] if pd.notna(r['delta_mean']) else float('nan'):+.4f}")
    else:
        print(f"  {f:<16}(absent)")

print()
assert a.loc["CONST_dead","status"]=="REJECT_HYGIENE"
print("  ok   constant feature rejected on hygiene")
assert a.loc["SPARSE_bad","status"]=="REJECT_HYGIENE"
print("  ok   20%-coverage feature rejected on hygiene")
assert a.loc["NOISE_dup","status"]=="DUPLICATE"
print("  ok   affine duplicate detected, not double-counted")
assert a.loc["SIGNAL_real","status"]=="KEEP", a.loc["SIGNAL_real","status"]
print("  ok   the ONE real signal survived to KEEP")
assert a.loc["NOISE_pure","status"]!="KEEP"
print(f"  ok   pure noise -> {a.loc['NOISE_pure','status']} (never reached KEEP)")
print(f"       NOT_TESTED means it ranked below the candidate cut - the cheap")
print(f"       filters removed it before the expensive stage, which is the point")
assert a.loc["REDUNDANT_copy","status"] in ("REDUNDANT_GLOBAL","NO_INCREMENTAL_VALUE","NOT_SELECTED_ANY_FOLD"), a.loc["REDUNDANT_copy","status"]
print("  ok   near-copy of the signal flagged REDUNDANT, not KEEP")

ap=json.loads((pdir/"audit"/"approved_features.json").read_text())
print(f"\n  approved: {ap['approved']}")
assert "SIGNAL_real" in ap["approved"]
assert "NOISE_pure" not in ap["approved"]
print("  ok   approved set contains the signal and not the noise")

rej=pd.read_csv(pdir/"audit"/"feature_rejections.csv")
print(f"  rejections retained: {len(rej)} rows with reasons")
assert len(rej)>0
print("  ok   rejected features kept for later reconsideration")
print("\nFEATURE AUDIT VERIFIED")

print("\n=== NULL FLOOR REJECTS EVERY PLANTED NOISE FEATURE ===")
ap2=json.loads((pdir/"audit"/"approved_features.json").read_text())
noise=["NOISE_pure","NOISE_dup","CONST_dead","SPARSE_bad","close","D_atr14",
       "ix_location_volume","ix_momentum_volume","REDUNDANT_copy"]
leaked=[f for f in noise if f in ap2["approved"]]
print(f"  null method: {ap2['null_method']}")
print(f"  null floor:  {ap2['null_floor_auc']:+.4f} AUC")
print(f"  approved:    {ap2['approved']}")
print(f"  noise that leaked through: {leaked or 'NONE'}")
assert not leaked, leaked
assert ap2["approved"]==ap2["base"]+["SIGNAL_real"], ap2["approved"]
print("  ok   approved set is exactly base + the one real signal")
print("       (a gaussian null let 3 noise features through; permutation did not)")
print("\nFEATURE AUDIT VERIFIED (permutation null)")


print("\n=== SELECTION IS FOLD-LOCAL ===")
sel=pd.read_csv(pdir/"audit"/"fold_selection.csv")
det=pd.read_csv(pdir/"audit"/"fold_selection_detail.csv")
print(f"  {det['fold'].nunique()} folds each chose candidates independently")
print(f"  {len(sel)} distinct features were selected in at least one fold")
top=sel.sort_values(["folds_selected","mean_train_ic"],ascending=False).head(6)
print(f"\n  {'feature':<22}{'folds':>7}{'survival':>10}{'train IC':>10}")
for _,r in top.iterrows():
    print(f"  {r['feature']:<22}{int(r['folds_selected'])}/{int(r['folds_total'])}"
          f"{r['survival']:>9.0%}{r['mean_train_ic']:>10.4f}")
sr=sel.set_index("feature")
assert sr.loc["SIGNAL_real","survival"]==1.0
print("\n  ok   the real signal was chosen independently in EVERY fold")
mixed=sel[(sel.survival>0)&(sel.survival<1)]
print(f"  {len(mixed)} features selected in some folds but not others")
print("       - that split is invisible to a single global ranking")
print("\nFEATURE AUDIT VERIFIED (fold-local selection)")
