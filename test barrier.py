"""Does barrier_scan resolve ORDER, or just extremes?"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0,".")
import barrier_scan as B

print("=== 1. ORDER IS RESOLVED, NOT APPROXIMATED ===")
# Two paths with IDENTICAL MFE/MAE, opposite bracket outcome.
hz=5
# A: +6% day1, -4% day4  -> TP first
runhiA=np.array([[.06,.06,.06,.06,.06]]); runloA=np.array([[0,0,0,-.04,-.04]])
# B: -4% day1, +6% day4  -> SL first
runhiB=np.array([[0,0,0,.06,.06]]);       runloB=np.array([[-.04,-.04,-.04,-.04,-.04]])
for name,rh,rl,want in [("TP first",runhiA,runloA,1),("SL first",runhiB,runloB,0)]:
    k_tp,k_sl=B.first_touch(rh,rl,np.array([.05]),np.array([.03]))
    got=int(k_tp[0]<k_sl[0])
    print(f"  {name:<9} MFE=+6.0% MAE=-4.0% | k_tp={k_tp[0]} k_sl={k_sl[0]} -> "
          f"TP first={bool(got)}")
    assert got==want
print("  ok   identical extremes, order correctly distinguished")
print("       (v1 called both of these ambiguous and scored them as losses)\n")

print("=== 2. SAME-DAY TIE REPORTED AS A RANGE, NOT A GUESS ===")
rh=np.array([[.06,.06,.06,.06,.06]]); rl=np.array([[-.04,-.04,-.04,-.04,-.04]])
fr=np.array([0.0]); atrp=np.array([0.03])
r=B.scan(rh,rl,fr,atrp,cost_bps=0.0)
row=r.loc[(r.tp=="0.05")&(r.sl=="0.030")].iloc[0]
print(f"  both barriers hit on day 1: lower={row['tp_first_lo']:.0%} "
      f"upper={row['tp_first_hi']:.0%} ambig={row['ambig']:.0%}")
assert row["tp_first_lo"]==0.0 and row["tp_first_hi"]==1.0 and row["ambig"]==1.0
print("  ok   the unknowable case spans 0%-100% instead of pretending\n")

print("=== 3. UNRESOLVED EPISODES ENTER THE PAYOFF ===")
# one episode that touches neither barrier and drifts +2%
rh=np.array([[.01,.015,.02,.02,.02]]); rl=np.array([[-.005,-.005,-.01,-.01,-.01]])
r=B.scan(rh,rl,np.array([0.02]),np.array([0.03]),cost_bps=0.0)
row=r.loc[(r.tp=="0.05")&(r.sl=="0.030")].iloc[0]
print(f"  neither={row['neither']:.0%}  pay_lo={row['pay_lo']:+.4f} "
      f"(realised +2.00%, zero cost)")
assert row["neither"]==1.0 and abs(row["pay_lo"]-0.02)<1e-9
rc=B.scan(rh,rl,np.array([0.02]),np.array([0.03]),cost_bps=35.0)
rowc=rc.loc[(rc.tp=="0.05")&(rc.sl=="0.030")].iloc[0]
print(f"  same episode at 35bps cost: pay_lo={rowc['pay_lo']:+.4f}")
assert abs(rowc["pay_lo"]-(0.02-0.0035))<1e-9
print("  ok   costs are subtracted, not asserted in a comment")
print("  ok   the 'none' population contributes its realised return")
print("       (v1's comment claimed this; v1's code did not do it)\n")

print("=== 4. TIMING IS REPORTED ===")
rh=np.array([[.06,.06,.06,.06,.06],[0,0,0,0,.06]])
rl=np.array([[0,0,0,0,0],[0,0,0,0,0]])
r=B.scan(rh,rl,np.array([.0,.0]),np.array([.03,.03]),cost_bps=0.0)
row=r.loc[(r.tp=="0.05")&(r.sl=="0.030")].iloc[0]
print(f"  one hits day 1, one hits day 5: median days={row['med_days_tp']:.1f} "
      f"P(TP<=1d)={row['tp_1d']:.0%}")
assert row["tp_1d"]==0.5
print("  ok   a fast target and a slow one are distinguishable\n")

print("=== 5. ATR SCALING CHANGES THE QUESTION ===")
# same 5% move; one stock is low-vol, one high-vol
rh=np.array([[.05,.05,.05,.05,.05]]*2); rl=np.zeros((2,5))
atrp=np.array([0.015,0.06])   # 5% = 3.3 ATR vs 0.83 ATR
fixed=B.scan(rh,rl,np.zeros(2),atrp,atr=False,cost_bps=0.0)
scaled=B.scan(rh,rl,np.zeros(2),atrp,atr=True,cost_bps=0.0)
f=fixed.loc[(fixed.tp=="0.05")].iloc[0]
s=scaled.loc[(scaled.tp=="1.00a")].iloc[0]
print(f"  fixed 5%   -> tp_touch {f['tp_touch']:.0%} (both stocks)")
print(f"  1.0 x ATR  -> tp_touch {s['tp_touch']:.0%} (only the low-vol one)")
assert f["tp_touch"]==1.0 and s["tp_touch"]==0.5
print("  ok   fixed-% treats a 3.3-ATR move and a 0.8-ATR move as the same event")
print("\nBARRIER SCAN VERIFIED")
