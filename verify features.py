"""
verify_features.py - checks features_daily against INDEPENDENT implementations.

The point is not to re-run the same code and see it agree with itself. Every
reference below is written from the textbook definition, mostly as an explicit
Python loop, so an error in the vectorised version cannot hide behind the same
mistake in the checker.

Three kinds of check:
  1. REFERENCE  - hand-rolled loop vs the module's vectorised output
  2. INVARIANT  - properties that must hold for any input (scale invariance,
                  bounds, known limits on degenerate series)
  3. SILENCE    - columns that came out entirely NaN, which is how a broken
                  computation disguises itself as missing data
"""
import sys, types, math
import numpy as np, pandas as pd

for n in ("kiteconnect", "kiteconnect.exceptions"):
    sys.modules[n] = types.ModuleType(n)
sys.modules["kiteconnect"].KiteConnect = object
for n in ("KiteException","TokenException","InputException","NetworkException",
          "DataException","GeneralException"):
    setattr(sys.modules["kiteconnect.exceptions"], n, type(n, (Exception,), {}))

import importlib.util
sp = importlib.util.spec_from_file_location("fd", "features_daily.py")
fd = importlib.util.module_from_spec(sp); sys.modules["fd"] = fd
sp.loader.exec_module(fd)

FAIL = []
def check(name, a, b, tol=1e-9, skip=0):
    a = pd.Series(a).astype(float).to_numpy()[skip:]
    b = pd.Series(b).astype(float).to_numpy()[skip:]
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() == 0:
        FAIL.append(f"{name}: nothing comparable"); print(f"  FAIL {name}: no overlap"); return
    d = np.nanmax(np.abs(a[m] - b[m]))
    ok = d <= tol
    print(f"  {'ok  ' if ok else 'FAIL'} {name:<28} maxdiff={d:.2e}  n={m.sum()}")
    if not ok: FAIL.append(f"{name}: maxdiff {d:.2e}")

def expect(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    if not cond: FAIL.append(name)

# ---------------- synthetic data ----------------
rng = np.random.default_rng(20260921)
N = 600
ret = rng.standard_t(4, N) * 0.013 + 0.0005   # fat tails: +5% days occur
close = 250 * np.exp(np.cumsum(ret))
open_ = close * (1 + rng.normal(0, 0.004, N))
high = np.maximum(close, open_) * (1 + np.abs(rng.normal(0, 0.007, N)))
low = np.minimum(close, open_) * (1 - np.abs(rng.normal(0, 0.007, N)))
vol = rng.lognormal(13.6, 0.85, N)            # volume is lognormal, not uniform
ts = pd.date_range("2022-01-03", periods=N, freq="B", tz=fd.IST)
raw = pd.DataFrame(dict(timestamp=ts, open=open_, high=high, low=low,
                        close=close, volume=vol))

C = pd.Series(close); H = pd.Series(high); L = pd.Series(low)
O = pd.Series(open_); V = pd.Series(vol)

print("\n=== 1. REFERENCE IMPLEMENTATIONS (explicit loops) ===")

# --- Wilder RSI, textbook loop ---
def ref_rsi(c, p):
    out = np.full(len(c), np.nan)
    d = np.diff(c, prepend=np.nan)
    g = np.where(d > 0, d, 0.0); l = np.where(d < 0, -d, 0.0)
    ag = al = np.nan
    for i in range(1, len(c)):
        if i < p:
            continue
        if i == p:
            ag = np.mean(g[1:p+1]); al = np.mean(l[1:p+1])
        else:
            ag = (ag*(p-1) + g[i]) / p
            al = (al*(p-1) + l[i]) / p
        if al == 0:   out[i] = 100.0 if ag > 0 else 50.0
        elif ag == 0: out[i] = 0.0
        else:         out[i] = 100 - 100/(1 + ag/al)
    return out
# pandas ewm(adjust=False) seeds from the first value, not an SMA, so the two
# converge rather than match exactly; compare after the seed washes out.
# pandas ewm(adjust=False) seeds from the first observation; Wilder seeds from
# an SMA of the first p values. The two CONVERGE geometrically rather than
# matching bit-for-bit, so the only honest comparison is at the warm-up
# boundary, where the seed has washed out. Measured decay for RSI-14:
#   bar  60 -> 1.8e-01 | bar 120 -> 8.5e-04 | bar 312 -> 1.3e-09
# WARMUP_BARS is 312, so every value that reaches a model is exact.
SEED_SKIP = fd.WARMUP_BARS
check("RSI-14 (Wilder)", ref_rsi(close, 14), fd._rsi(C, 14), tol=1e-7, skip=SEED_SKIP)
check("RSI-7 (Wilder)",  ref_rsi(close, 7),  fd._rsi(C, 7),  tol=1e-7, skip=SEED_SKIP)

# --- True Range, textbook ---
def ref_tr(h, l, c):
    out = np.full(len(c), np.nan)
    out[0] = h[0] - l[0]
    for i in range(1, len(c)):
        out[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    return out
tr_ref = ref_tr(high, low, close)
check("True Range", tr_ref, fd._true_range(H, L, C))

# --- ATR (Wilder) ---
def ref_atr(h, l, c, p):
    tr = ref_tr(h, l, c); out = np.full(len(c), np.nan); a = np.nan
    for i in range(len(c)):
        if i < p-1: continue
        if i == p-1: a = np.nanmean(tr[:p])
        else: a = (a*(p-1) + tr[i]) / p
        out[i] = a
    return out
check("ATR-14 (Wilder)", ref_atr(high, low, close, 14), fd._atr(H, L, C, 14),
      tol=1e-7, skip=SEED_SKIP)

# --- ADX / DI, textbook loop ---
def ref_adx(h, l, c, p):
    n = len(c)
    pdm = np.zeros(n); mdm = np.zeros(n)
    for i in range(1, n):
        up = h[i]-h[i-1]; dn = l[i-1]-l[i]
        pdm[i] = up if (up > dn and up > 0) else 0.0
        mdm[i] = dn if (dn > up and dn > 0) else 0.0
    tr = ref_tr(h, l, c)
    atr = np.full(n, np.nan); sp = np.full(n, np.nan); sm = np.full(n, np.nan)
    a = pp = mm = np.nan
    for i in range(n):
        if i < p-1: continue
        if i == p-1:
            a = np.nanmean(tr[:p]); pp = np.nanmean(pdm[:p]); mm = np.nanmean(mdm[:p])
        else:
            a = (a*(p-1)+tr[i])/p; pp = (pp*(p-1)+pdm[i])/p; mm = (mm*(p-1)+mdm[i])/p
        atr[i], sp[i], sm[i] = a, pp, mm
    pdi = 100*sp/atr; mdi = 100*sm/atr
    dx = 100*np.abs(pdi-mdi)/(pdi+mdi)
    adx = np.full(n, np.nan); ax = np.nan
    for i in range(n):
        if np.isnan(dx[i]): continue
        ax = dx[i] if np.isnan(ax) else (ax*(p-1)+dx[i])/p
        adx[i] = ax
    return adx, pdi, mdi
ra, rp, rm = ref_adx(high, low, close, 14)
ma, mp_, mm_ = fd._adx(H, L, C, 14)
check("+DI 14", rp, mp_, tol=1e-7, skip=SEED_SKIP)
check("-DI 14", rm, mm_, tol=1e-7, skip=SEED_SKIP)
check("ADX 14", ra, ma, tol=1e-7, skip=SEED_SKIP)

# --- SMA / EMA ---
ref_sma = np.array([np.mean(close[i-19:i+1]) if i >= 19 else np.nan for i in range(N)])
check("SMA-20", ref_sma, fd._sma(C, 20))
def ref_ema(c, span):
    a = 2/(span+1); out = np.full(len(c), np.nan); e = c[0]
    for i in range(len(c)):
        e = c[i] if i == 0 else a*c[i] + (1-a)*e
        if i >= span-1: out[i] = e
    return out
check("EMA-20", ref_ema(close, 20), fd._ema(C, 20), tol=1e-9)

# --- expanding percentile rank, brute force ---
def ref_xrank(x, mp=60):
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        w = x[:i+1]; w = w[~np.isnan(w)]
        if len(w) < mp: continue
        v = x[i]
        lo = np.sum(w < v); eq = np.sum(w == v)
        out[i] = (lo + (eq+1)/2) / len(w)
    return out
check("expanding rank pct", ref_xrank(close), fd._expand_rank_pct(C), tol=1e-12)

# --- rolling OLS slope vs polyfit ---
def ref_slope(y, w):
    out = np.full(len(y), np.nan)
    t = np.arange(w, dtype=float)
    for i in range(w-1, len(y)):
        out[i] = np.polyfit(t, y[i-w+1:i+1], 1)[0]
    return out
check("rolling OLS slope(20)", ref_slope(close, 20),
      fd._rolling_ols_slope_fast(C, 20), tol=1e-7)

print("\n=== 2. FULL PIPELINE vs REFERENCE ===")
out = fd.compute_daily_indicators(raw.copy())

check("D_atr14", ref_atr(high, low, close, 14), out["D_atr14"], 1e-7, skip=SEED_SKIP)
check("D_rsi14", ref_rsi(close, 14), out["D_rsi14"], 1e-7, skip=SEED_SKIP)
check("D_adx14", ra, out["D_adx14"], 1e-7, skip=SEED_SKIP)

# OBV
def ref_obv(c, v):
    o = np.zeros(len(c))
    for i in range(1, len(c)):
        o[i] = o[i-1] + (v[i] if c[i] > c[i-1] else (-v[i] if c[i] < c[i-1] else 0))
    return o
check("D_obv", ref_obv(close, vol), out["D_obv"])

# CMF-20
mfm = ((close-low)-(high-close))/(high-low)
mfv = mfm*vol
ref_cmf = np.array([mfv[i-19:i+1].sum()/vol[i-19:i+1].sum() if i>=19 else np.nan
                    for i in range(N)])
check("D_cmf20", ref_cmf, out["D_cmf20"])

# Bollinger %B (sample std, ddof=1)
sma20 = ref_sma
std20 = np.array([np.std(close[i-19:i+1], ddof=1) if i>=19 else np.nan for i in range(N)])
ref_pctb = (close - (sma20-2*std20)) / ((sma20+2*std20)-(sma20-2*std20))
check("D_bb_pctB_20", np.clip(ref_pctb,-5,5), out["D_bb_pctB_20"], 1e-9)

# Donchian position (prior 20 bars, excludes today)
hi20 = np.array([np.max(high[i-20:i]) if i>=20 else np.nan for i in range(N)])
lo20 = np.array([np.min(low[i-20:i]) if i>=20 else np.nan for i in range(N)])
check("D_donch_pos_20", np.clip((close-lo20)/(hi20-lo20),-1,2),
      out["D_donch_pos_20"], 1e-9)

# 52w high distance (prior 252 bars)
hi52 = np.array([np.max(high[i-252:i]) if i>=252 else np.nan for i in range(N)])
atr14 = ref_atr(high, low, close, 14)
check("D_dist_from_52wh", (close-hi52)/atr14, out["D_dist_from_52wh"], 1e-7, skip=SEED_SKIP)

# Pivots
check("D_cpr_pivot", (high+low+close)/3, out["D_cpr_pivot"])
check("D_cpr_bc", (high+low)/2, out["D_cpr_bc"])
check("D_cpr_tc", 2*(high+low+close)/3-(high+low)/2, out["D_cpr_tc"])
check("D_gap_pct", (open_-np.roll(close,1))/np.roll(close,1)*100, out["D_gap_pct"], 1e-9, skip=1)

# Yang-Zhang: variance must be positive and scale-free
yz = pd.to_numeric(out["D_vol_yz_20"], errors="coerce")
expect("D_vol_yz_20 non-negative", bool((yz.dropna() >= 0).all()))

print("\n=== 3. INVARIANTS ===")
# scale invariance: multiply ALL prices by k -> bounded oscillators unchanged
k = 7.3
scaled = raw.copy()
for c_ in ("open","high","low","close"): scaled[c_] = scaled[c_]*k
o2 = fd.compute_daily_indicators(scaled)
for col, mode in [("D_rsi14","same"), ("D_adx14","same"), ("D_bb_pctB_20","same"),
                  ("D_donch_pos_20","same"), ("D_atr14","scale"), ("D_atr_pct","same"),
                  ("D_range_pct","same"), ("D_gap_pct","same")]:
    a = pd.to_numeric(out[col], errors="coerce")
    b = pd.to_numeric(o2[col], errors="coerce")
    b_adj = b/k if mode == "scale" else b
    d = np.nanmax(np.abs(a-b_adj))
    expect(f"{col} scale-{'equivariant' if mode=='scale' else 'invariant'}",
           d < 1e-6, f"maxdiff={d:.2e}")

# degenerate series
flat = pd.DataFrame(dict(timestamp=ts, open=100.0, high=100.0, low=100.0,
                         close=100.0, volume=1e5))
of = fd.compute_daily_indicators(flat)
expect("flat series -> ATR 0", float(np.nanmax(np.abs(
    pd.to_numeric(of["D_atr14"], errors="coerce").dropna()))) < 1e-12)
expect("flat series -> RSI 50", bool(
    (pd.to_numeric(of["D_rsi14"], errors="coerce").dropna() == 50).all()))

mono = close.copy(); mono = np.sort(mono)
mdf = pd.DataFrame(dict(timestamp=ts, open=mono, high=mono*1.001, low=mono*0.999,
                        close=mono, volume=1e5))
om = fd.compute_daily_indicators(mdf)
expect("monotone up -> RSI 100", float(pd.to_numeric(
    om["D_rsi14"], errors="coerce").dropna().iloc[-1]) == 100.0)

# bounded columns really are bounded
for col, lo_, hi_ in [("D_rsi14",0,100), ("D_rsi7",0,100), ("D_adx14",0,100),
                      ("D_pdi14",0,100), ("D_mdi14",0,100),
                      ("D_donch_pos_20",-1,2), ("D_bb_pctB_20",-5,5)]:
    v_ = pd.to_numeric(out[col], errors="coerce").dropna()
    expect(f"{col} within [{lo_},{hi_}]", bool(v_.between(lo_,hi_).all()),
           f"range=[{v_.min():.2f},{v_.max():.2f}]")

print("\n=== 4. SILENT FAILURES (all-NaN / constant columns) ===")
fin = fd.finalize_for_cache(out)
val = fin.iloc[320:]
allnan, const = [], []
for c_ in val.columns:
    if c_ in ("timestamp",): continue
    s_ = pd.to_numeric(val[c_], errors="coerce")
    if s_.isna().all(): allnan.append(c_)
    elif s_.nunique(dropna=True) <= 1: const.append(c_)
print(f"  all-NaN after warm-up : {allnan or 'none'}")
print(f"  constant after warm-up: {const or 'none'}")
if allnan: FAIL.append(f"all-NaN columns: {allnan}")

print("\n=== 4b. v26 ADDITIONS ===")
lo52 = np.array([np.min(low[i-252:i]) if i>=252 else np.nan for i in range(N)])
check("D_dist_from_52wl", (close-lo52)/atr14, out["D_dist_from_52wl"], 1e-7, skip=SEED_SKIP)
peak = np.array([np.max(close[i-251:i+1]) if i>=251 else np.nan for i in range(N)])
check("D_drawdown_252", close/peak-1, out["D_drawdown_252"], 1e-12)
check("D_intraday_ret_pct", (close-open_)/open_*100, out["D_intraday_ret_pct"], 1e-12)
rr = np.diff(close, prepend=np.nan)/np.roll(close,1); rr[0]=np.nan
ref_rv20 = np.array([np.std(rr[i-19:i+1], ddof=1)*np.sqrt(252)*100 if i>=20 else np.nan
                     for i in range(N)])
check("D_realvol_20", ref_rv20, out["D_realvol_20"], 1e-9, skip=25)
bigup = (rr >= 0.05).astype(float)
ref_n252 = np.array([np.nansum(bigup[i-251:i+1]) if i>=251 else np.nan for i in range(N)])
check("D_n_5pct_up_252", ref_n252, out["D_n_5pct_up_252"], 1e-12)
dd = pd.to_numeric(out["D_drawdown_252"], errors="coerce").dropna()
expect("D_drawdown_252 <= 0", bool((dd <= 1e-12).all()), f"max={dd.max():.2e}")
expect("D_days_since_5pct_up >= 0",
       bool((pd.to_numeric(out["D_days_since_5pct_up"]).dropna() >= 0).all()))
nz = int((pd.to_numeric(out["D_n_5pct_up_252"], errors="coerce").fillna(0) > 0).sum())
expect("+5% days actually occur in the sample", nz > 0, f"{nz} bars with a prior +5% day")

print("\n=== 4c. PATHOLOGICAL INPUTS ===")
# The cache quarantines bad data, but the feature engine must still degrade
# predictably rather than emit a confident wrong number. What we require:
# NO exception, NO inf, and NO fabricated value where the input was unknown.
def pathological(label, frame, expect_raise=False):
    try:
        o_ = fd.compute_daily_indicators(frame.copy())
        o_ = fd.finalize_for_cache(o_)
    except Exception as e:
        verdict = "raised" if expect_raise else f"UNEXPECTED {type(e).__name__}: {e}"
        print(f"  {'ok  ' if expect_raise else 'FAIL'} {label:<26} {verdict[:60]}")
        if not expect_raise: FAIL.append(f"{label}: {type(e).__name__}")
        return None
    num = o_.select_dtypes(include=[np.number])
    n_inf = int(np.isinf(num.to_numpy(dtype="float64", na_value=np.nan)).sum())
    print(f"  {'ok  ' if n_inf==0 else 'FAIL'} {label:<26} cols={o_.shape[1]} inf={n_inf}")
    if n_inf: FAIL.append(f"{label}: {n_inf} inf values")
    return o_

base = raw.copy()

p = base.copy(); p.loc[100:104, ["open","high","low","close"]] = np.nan
pathological("NaN OHLC block", p)

p = base.copy(); p.loc[:, "volume"] = 0.0
o_zero = pathological("zero volume throughout", p)

p = base.copy(); p.loc[:, "volume"] = np.nan
o_nov = pathological("volume all NaN (index)", p)
if o_nov is not None:
    for c_ in ("D_obv","D_dollar_vol","D_cmf20","D_amihud_20"):
        v_ = pd.to_numeric(o_nov[c_], errors="coerce")
        expect(f"  {c_} not fabricated", bool(v_.isna().all()),
               f"zeros={int((v_==0).sum())}")

p = base.copy()
for _c in ("open","high","low"):
    p.loc[200:210, _c] = p.loc[200:210, "close"].to_numpy()
pathological("zero high-low range", p)

p = base.copy(); p.loc[300:, ["open","high","low","close"]] /= 5.0
pathological("unadjusted 1:5 split", p)

p = base.copy(); p.loc[400, "close"] *= 50; p.loc[400, "high"] = p.loc[400,"close"]
pathological("extreme outlier bar", p)

p = base.copy(); p.loc[:, "volume"] = 1e6
pathological("constant volume", p)

p = base.copy(); p.loc[50, ["open","high","low","close"]] = -10.0
pathological("negative prices", p)

p = pd.concat([base, base.iloc[[10]]]).sort_values("timestamp").reset_index(drop=True)
pathological("duplicate timestamp", p)

p = base.drop(index=range(120,160)).reset_index(drop=True)
pathological("40 missing sessions", p)

pathological("single row", base.iloc[[0]].reset_index(drop=True))
pathological("two rows", base.iloc[:2].reset_index(drop=True))

print("\n=== 4d. STATIC LEAK CHECKER ===")
try:
    fd.static_leak_check(); expect("static checker clean on this module", True)
except RuntimeError as e:
    expect("static checker clean on this module", False, str(e)[:90])

print("\n=== 5. DUPLICATE COLUMNS ===")
num = val.select_dtypes(include=[np.number])
dupes = []
cols = list(num.columns)
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        a = num[cols[i]].to_numpy(dtype=float); b = num[cols[j]].to_numpy(dtype=float)
        m = ~(np.isnan(a)|np.isnan(b))
        if np.nanstd(a[m]) == 0 or np.nanstd(b[m]) == 0:
            continue    # constant on this sample proves nothing about duplication
        if m.sum() > 50 and np.allclose(a[m], b[m], rtol=0, atol=1e-12):
            dupes.append((cols[i], cols[j]))
print(f"  identical pairs: {dupes or 'none'}")
if dupes: FAIL.append(f"duplicate columns: {dupes}")

print("\n" + "="*60)
if FAIL:
    print(f"{len(FAIL)} PROBLEM(S):")
    for f_ in FAIL: print("   -", f_)
else:
    print("ALL VERIFICATION CHECKS PASSED")
print("="*60)
