import sys, json, tempfile, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0,".")
import reconcile as rc

root=Path(tempfile.mkdtemp())
ses=pd.DatetimeIndex(pd.bdate_range("2023-01-02","2024-12-31"))
def mk(seed,ix):
    r=np.random.default_rng(seed); c=100*np.exp(np.cumsum(r.normal(0,.014,len(ix))))
    return pd.DataFrame({"timestamp":ix,"open":c,"high":c*1.01,"low":c*.99,"close":c,
                         "volume":r.lognormal(13.5,.8,len(ix))})
TRUTH={}
def w(sym,df):
    df.to_parquet(root/f"{sym}_daily.parquet",index=False)
    (root/f"{sym}_daily.ok.json").write_text(json.dumps({
        "schema_version":"v27","rows":len(df),"series_kind":"equity",
        "last_timestamp":str(df['timestamp'].iloc[-1])},default=str))

# CLEAN: cache and vendor agree; vendor sits at a different level (adjustment basis)
clean=mk(1,ses); w("CLEAN",clean); TRUTH["CLEAN"]=clean.assign(
    **{c:clean[c]*3.7 for c in ("open","high","low","close")})
# SPLIT: vendor applied a 1:5 the cache did not
sp=mk(2,ses); w("SPLITMISS",sp)
v=sp.copy(); v.loc[300:,["open","high","low","close"]]/=5.0; TRUTH["SPLITMISS"]=v
# BADPRINT: one wrong close in the cache
bp=mk(3,ses); ref=bp.copy(); bp.loc[400,"close"]*=1.2; w("BADPRINT",bp); TRUTH["BADPRINT"]=ref
# GAPPY: cache missing 20 sessions the vendor has
g=mk(4,ses); w("GAPPY",g.drop(index=range(100,120)).reset_index(drop=True)); TRUTH["GAPPY"]=g

def fake_fetcher(symbol,start,end):
    d=TRUTH.get(symbol)
    if d is None: return pd.DataFrame()
    m=(d["timestamp"]>=pd.Timestamp(start))&(d["timestamp"]<=pd.Timestamp(end))
    return d.loc[m].copy()

rep=rc.reconcile(root,["CLEAN","SPLITMISS","BADPRINT","GAPPY"],
                 fetcher=fake_fetcher,verbose=False)
res={r["symbol"]:r for r in rep["results"]}
for k,r in res.items():
    print(f"  {k:<12} {r['status']:<18} rate={r.get('mismatch_rate')} "
          f"maxdiff={r.get('max_return_diff')} missing={r.get('n_sessions_missing_from_cache')}")

assert res["CLEAN"]["status"]=="OK", "level difference must NOT be an error"
print("\n  ok   vendor at a different adjustment level -> OK (returns compared, not levels)")
assert res["SPLITMISS"]["corporate_action_suspects"], "missed split must be flagged"
assert res["SPLITMISS"]["status"]!="OK", "a missed split cannot pass"
print(f"  ok   missed 1:5 split flagged: {res['SPLITMISS']['corporate_action_suspects'][0]}")
assert res["BADPRINT"]["status"] in ("MISMATCH","CORPORATE_ACTION"), res["BADPRINT"]["status"]
print(f"  ok   single bad print caught: {res['BADPRINT']['worst_days'][0]}")
assert res["GAPPY"]["status"]=="MISSING_SESSIONS" and res["GAPPY"]["n_sessions_missing_from_cache"]==20
print(f"  ok   20 sessions missing from cache detected")
print("\nRECONCILIATION LOGIC VERIFIED")
