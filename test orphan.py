import sys, types, datetime as dt
import numpy as np, pandas as pd
for n in ("kiteconnect","kiteconnect.exceptions"): sys.modules[n]=types.ModuleType(n)
sys.modules["kiteconnect"].KiteConnect=object
for n in ("KiteException","TokenException","InputException","NetworkException","DataException","GeneralException"):
    setattr(sys.modules["kiteconnect.exceptions"],n,type(n,(Exception,),{}))
import importlib.util
sp=importlib.util.spec_from_file_location("dc","Daily_cache_v27.py"); dc=importlib.util.module_from_spec(sp)
sys.modules["dc"]=dc; sp.loader.exec_module(dc)

def mk(ix):
    c=100*np.exp(np.cumsum(np.random.default_rng(1).normal(0,.01,len(ix))))
    return pd.DataFrame({"timestamp":ix,"open":c,"high":c*1.01,"low":c*.99,
                         "close":c,"volume":1e5})

print("=== DELHIVERY-shaped: 8 orphan bars in 2016, real series from May 2022 ===")
orph=pd.DatetimeIndex(["2016-01-18","2016-02-01","2016-03-08","2016-04-12",
                       "2016-06-02","2016-07-19","2016-09-05","2016-11-21"])
real=pd.DatetimeIndex(pd.bdate_range("2022-05-24","2026-09-21"))
df=mk(orph.append(real).sort_values())
out,dropped=dc.trim_orphan_bars(df)
print(f"  in={len(df)} out={len(out)} dropped={len(dropped)}")
print(f"  new first bar: {pd.to_datetime(out.timestamp.iloc[0]).date()}")
assert len(dropped)==8 and pd.to_datetime(out.timestamp.iloc[0]).date()==dt.date(2022,5,24)
print("  ok   all 8 orphans removed, real history intact\n")

print("=== clean series: nothing must be trimmed ===")
clean=mk(pd.DatetimeIndex(pd.bdate_range("2015-01-01","2026-09-21")))
o2,d2=dc.trim_orphan_bars(clean)
print(f"  in={len(clean)} out={len(o2)} dropped={len(d2)}")
assert len(d2)==0 and len(o2)==len(clean)
print("  ok   untouched\n")

print("=== genuinely illiquid smallcap (trades ~60% of sessions, no orphans) ===")
rng=np.random.default_rng(7)
full=pd.DatetimeIndex(pd.bdate_range("2018-01-01","2026-09-21"))
keep=full[rng.random(len(full))<0.60]
thin=mk(keep)
o3,d3=dc.trim_orphan_bars(thin)
print(f"  in={len(thin)} out={len(o3)} dropped={len(d3)}")
assert len(d3)==0, "thin-but-continuous must NOT be trimmed as orphans"
print("  ok   thin liquidity is not mistaken for orphan bars\n")

print("=== late lister with NO orphans ===")
late=mk(pd.DatetimeIndex(pd.bdate_range("2023-03-01","2026-09-21")))
o4,d4=dc.trim_orphan_bars(late)
assert len(d4)==0
print(f"  ok   late listing left alone (first bar {pd.to_datetime(o4.timestamp.iloc[0]).date()})\n")

print("=== real cache shapes (FUSION / ELIN / KOVAI / GOODYEAR) ===")
def build(spec):
    out=[]
    for yr,n in spec.items():
        days=pd.bdate_range(f"{yr}-01-01",f"{yr}-12-31"); step=max(1,len(days)//max(n,1))
        out.extend(days[::step][:n])
    return pd.DatetimeIndex(sorted(out))
real={
 "FUSION":({2016:123,2022:34,2023:246,2024:249,2025:249,2026:178},123),
 "ELIN":({2015:183,2022:1,2023:246,2024:249,2025:249,2026:178},183),
 "KOVAI":({2015:1,2016:211,2017:223,2021:102,2022:248,2023:201,2024:249,2025:249,2026:107},435),
 "DELHIVERY":({2016:8,2022:153,2023:246,2024:249,2025:249,2026:178},8),
}
for sym,(spec,expect) in real.items():
    o,d=dc.trim_orphan_bars(mk(build(spec)))
    print(f"  {sym:<10} dropped={len(d):<4} expected={expect:<4} new first={pd.to_datetime(o.timestamp.iloc[0]).date()}")
    assert len(d)==expect, f"{sym}: {len(d)} != {expect}"
print("  ok   all pre-listing blocks removed regardless of size\n")

print("=== safety rail: never trim below a year of history ===")
tiny=mk(build({2015:300,2026:100}))
o,d=dc.trim_orphan_bars(tiny)
print(f"  300 old bars + 100 recent: dropped={len(d)} (would leave <{dc.ORPHAN_MIN_REMAINING})")
assert len(d)==0
print("  ok   refuses rather than gut the series\n")

print("ORPHAN DETECTION VERIFIED")
