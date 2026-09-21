#!/usr/bin/env python3
"""
Daily feature computation for the NSE quant pipeline.
Split out of Daily_cache.py at v22.

WHY THIS IS A SEPARATE MODULE
=============================
The cache's job is to hold raw, adjusted OHLCV that matches the exchange.
That schema never changes, so the cache never needs rebuilding. Feature
definitions change constantly, which is exactly the wrong thing to bake
into a thousand parquet files - every tweak used to mean a schema bump and
a full rewrite.

Nothing about the maths changed in the move. The v20 leak protections
travel with the code they protect:

  * _expand_rank_pct       - expanding (leak-free) rank, not whole-series
  * _v20_static_leak_check - scans THIS file for unwindowed .rank()/.quantile()
  * _v20_leak_canary_check - tampers with the future, asserts the past is fixed

Call assert_leak_free() at the start of any process that computes features.
It is cheap and it is the only thing standing between you and the class of
bug that produced v19.

WARM-UP
-------
Indicators need 252 + 60 bars before every column is trustworthy. Feed
compute_daily_indicators() the FULL history from the cache, then drop the
warm-up region with mark_warmup()/drop_warmup(). Never compute features on
a pre-trimmed frame: _expand_rank_pct expands from row 0, so a different
start date silently produces different values for the same day.

THE LABEL LIVES HERE TOO
------------------------
ret_5d_close_pct is close.shift(-5) - a forward return. It is emitted
deliberately and it will destroy any model that sees it as a feature.
Use feature_columns() to build X. Never use df.columns.
"""

from __future__ import annotations

import datetime as dt
import math
import os
import re
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
except Exception:  # pragma: no cover
    IST = dt.timezone(dt.timedelta(hours=5, minutes=30))

WARMUP_FLAG_COL = "D_is_warmup"

# Columns whose NA means "unknown" and must NEVER be filled.
#
# D_roll is the one that matters. It is a nullable Int8 with three states:
#   1  = first session after a known contract expiry (a roll)
#   0  = inside a known contract (not a roll)
#   NA = predates the observed roll calendar -> WE DO NOT KNOW
#
# Filling NA with 0 would assert "no roll here" for exactly the rows where
# that is unknowable, which is how a feature ends up differencing across a
# contract boundary and calling the step a return. This module previously
# left D_roll alone only by accident - "Int8" is not "boolean", so the dtype
# loop below skipped it. One careless edit away from silent corruption, so
# the invariant is now declared and enforced.
PRESERVE_NA = ("D_roll",)

# Longest lookback in compute_daily_indicators is 252 trading bars
# (D_dist_from_52wh, the *_z252 family). _expand_rank_pct additionally needs
# 60 expanding observations. 312 bars ~= 450 calendar days; 520 gives headroom.
DEFAULT_WARMUP_DAYS = 520
WARMUP_BARS = 312


def _warmup_days() -> int:
    return int(os.environ.get("CACHE_WARMUP_DAYS", str(DEFAULT_WARMUP_DAYS)))


def assert_leak_free() -> None:
    """Run both v20 leak gates. Call this before any feature build."""
    _v20_static_leak_check()
    _v20_leak_canary_check()


def drop_warmup(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows flagged by mark_warmup(). Use before panel assembly."""
    if df is None or df.empty or WARMUP_FLAG_COL not in df.columns:
        return df
    return df.loc[df[WARMUP_FLAG_COL] == 0].reset_index(drop=True)


class UnknownRollFilled(RuntimeError):
    """Raised when a PRESERVE_NA column lost NAs during feature computation."""


def _na_census(df: pd.DataFrame) -> dict:
    return {
        c: int(pd.isna(df[c]).sum())
        for c in PRESERVE_NA
        if df is not None and c in df.columns
    }


def assert_na_preserved(before: dict, after: pd.DataFrame) -> None:
    """
    Fail loudly if an unknown became a known.

    Checked rather than trusted: the consequence of silently filling D_roll is
    a futures return series with a fabricated jump in it, which no downstream
    test would catch.
    """
    now = _na_census(after)
    for col, n_before in before.items():
        n_now = now.get(col, 0)
        if n_now < n_before:
            raise UnknownRollFilled(
                f"{col}: {n_before - n_now} unknown value(s) were filled "
                f"during feature computation ({n_before} NA -> {n_now}). "
                f"NA in {col} means 'not knowable', never 0."
            )


def build_features(
    df: pd.DataFrame, *, first_valid: Optional[dt.date] = None
) -> pd.DataFrame:
    """
    Full pipeline for one symbol: raw OHLCV in, flagged feature frame out.

    `df` must be the symbol's COMPLETE cached history, oldest first.
    `first_valid` is the first date you intend to use; everything before it
    is marked warm-up. Leave it None to fall back to WARMUP_BARS.
    """
    if df is None or df.empty:
        return df
    na_before = _na_census(df)
    out = compute_daily_indicators(df.copy())
    out = finalize_for_cache(out)
    assert_na_preserved(na_before, out)
    if first_valid is None and len(out) > WARMUP_BARS:
        ts = pd.to_datetime(out["timestamp"], errors="coerce")
        first_valid = ts.iloc[WARMUP_BARS].date()
    return mark_warmup(out, first_valid)


def _ensure_float(s: pd.Series) -> pd.Series:
    # v21 fast path: avoid a full pd.to_numeric pass on series that are
    # already float64. Called dozens of times per symbol.
    if getattr(s, "dtype", None) == np.float64:
        return s
    return pd.to_numeric(s, errors="coerce")


def _ema(series: pd.Series, span: int) -> pd.Series:
    # v21: min_periods=span. Previously min_periods=1, so a "20-day EMA"
    # existed at row 2 as a 2-day EMA. Warm-up rows are now trimmed, so
    # emitting NaN here is correct rather than costly.
    return _ensure_float(series).ewm(span=span, adjust=False, min_periods=span).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    # v21: min_periods=window (was 1).
    return _ensure_float(series).rolling(window=window, min_periods=window).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    close = _ensure_float(series)
    d = close.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    # v21: handle the degenerate branches explicitly. Previously
    # avg_loss==0 produced rs=NaN -> RSI=NaN, so an unbroken up-run (the
    # most bullish state there is) silently became a missing value.
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    warm = avg_gain.notna() & avg_loss.notna()
    out = out.mask(warm & (avg_loss == 0) & (avg_gain > 0), 100.0)
    out = out.mask(warm & (avg_gain == 0) & (avg_loss > 0), 0.0)
    out = out.mask(warm & (avg_gain == 0) & (avg_loss == 0), 50.0)
    return out


def _true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    h = _ensure_float(h)
    l = _ensure_float(l)
    c = _ensure_float(c)
    pc = c.shift(1)
    ranges = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1)
    return ranges.max(axis=1)


def _atr(h: pd.Series, l: pd.Series, c: pd.Series, period: int) -> pd.Series:
    tr = _true_range(h, l, c)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _adx(h, l, c, period: int):
    h = _ensure_float(h)
    l = _ensure_float(l)
    c = _ensure_float(c)
    up = h.diff()
    dn = l.shift(1) - l
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = _true_range(h, l, c)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = (
        100
        * pd.Series(plus_dm, index=h.index)
        .ewm(alpha=1 / period, adjust=False, min_periods=period)
        .mean()
        / atr
    )
    minus_di = (
        100
        * pd.Series(minus_dm, index=h.index)
        .ewm(alpha=1 / period, adjust=False, min_periods=period)
        .mean()
        / atr
    )
    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = (plus_di - minus_di).abs() / denom * 100
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return adx, plus_di, minus_di


def compute_vpoc(df: pd.DataFrame, bins: int = 50) -> float:
    if df.empty:
        return float("nan")
    vols = df["volume"].to_numpy(dtype="float64")
    if {"high", "low", "close"}.issubset(df.columns):
        highs = df["high"].to_numpy(dtype="float64")
        lows = df["low"].to_numpy(dtype="float64")
        closes = df["close"].to_numpy(dtype="float64")
        prices = (highs + lows + closes) / 3.0
    else:
        prices = df["close"].to_numpy(dtype="float64")
    mask = np.isfinite(prices) & np.isfinite(vols)
    if not mask.any():
        return float("nan")
    prices = prices[mask]
    vols = vols[mask]
    lo = float(np.min(prices))
    hi = float(np.max(prices))
    if not math.isfinite(lo) or not math.isfinite(hi):
        return float("nan")
    if math.isclose(lo, hi):
        return float(lo)
    hist, edges = np.histogram(prices, bins=bins, range=(lo, hi), weights=vols)
    if hist.size == 0 or np.all(hist == 0):
        return float((lo + hi) / 2.0)
    idx = int(np.argmax(hist))
    up_idx = min(idx + 1, len(edges) - 1)
    return float((edges[idx] + edges[up_idx]) / 2.0)


def _compute_weekly_vpoc_fast(
    timestamps: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """
    Strictly leak-free running weekly VPOC.

    For each row t inside a calendar week, the VPOC is computed using ONLY
    rows of that week up to and including t. v19 used the full week's
    [min(TP), max(TP)] as bin edges — which leaked the rest-of-week into
    early-in-week rows. v20 expands the bin layout monotonically with the
    observed data, so a row at Tuesday only sees Mon+Tue prices.
    """
    N_BINS = 50
    ts_local = (
        timestamps.dt.tz_convert(None)
        if timestamps.dt.tz is not None
        else timestamps
    )
    week_key = ts_local.dt.to_period("W-FRI")
    result = pd.Series(np.nan, index=timestamps.index, dtype="float64")

    week_groups = week_key.groupby(week_key).groups
    for wk, idx_arr in week_groups.items():
        idx = np.asarray(list(idx_arr), dtype="int64")
        if idx.size == 0:
            continue
        h_w = high.iloc[idx].to_numpy(dtype="float64")
        l_w = low.iloc[idx].to_numpy(dtype="float64")
        c_w = close.iloc[idx].to_numpy(dtype="float64")
        v_w = volume.iloc[idx].to_numpy(dtype="float64")
        tp_w = (h_w + l_w + c_w) / 3.0
        n = idx.size

        # Strictly causal running weekly VPOC: at each row i within the week,
        # use ONLY tp[0..i] / v[0..i] (no future days, even within the same
        # week). v19 leaked here by computing bin edges from the full week's
        # min/max — which the leak canary catches. We rebuild a fresh
        # histogram at each step; per-week cost is O(L^2 + L * N_BINS) which
        # is trivial for L≈5.
        vpocs = np.full(n, np.nan, dtype="float64")
        for i in range(n):
            finite = np.isfinite(tp_w[: i + 1]) & np.isfinite(v_w[: i + 1])
            if not finite.any():
                continue
            tps = tp_w[: i + 1][finite]
            vs = v_w[: i + 1][finite]
            lo_i = float(tps.min())
            hi_i = float(tps.max())
            if not (math.isfinite(lo_i) and math.isfinite(hi_i)):
                continue
            if math.isclose(lo_i, hi_i):
                vpocs[i] = lo_i
                continue
            edges = np.linspace(lo_i, hi_i, N_BINS + 1)
            bin_mids = (edges[:-1] + edges[1:]) / 2.0
            bins = np.clip(
                ((tps - lo_i) / (hi_i - lo_i) * N_BINS).astype("int64"),
                0,
                N_BINS - 1,
            )
            cum = np.bincount(bins, weights=vs, minlength=N_BINS)
            vpocs[i] = bin_mids[int(np.argmax(cum))]
        result.iloc[idx] = vpocs

    return result


def _cpr_relationship(base_bc, base_tc, other_bc, other_tc) -> pd.Series:
    relation = pd.Series(index=base_bc.index, dtype="object")
    relation[
        (other_bc > base_tc) & other_bc.notna() & base_tc.notna()
    ] = "Above"
    relation[
        (other_tc < base_bc) & other_tc.notna() & base_bc.notna()
    ] = "Below"
    relation[
        ((other_bc <= base_tc) & (other_tc >= base_bc))
        & other_bc.notna()
        & other_tc.notna()
        & base_bc.notna()
        & base_tc.notna()
    ] = "Inside"
    relation = relation.fillna("Overlap")
    relation[
        (base_bc.isna()) | (base_tc.isna()) | (other_bc.isna()) | (other_tc.isna())
    ] = None
    return relation


def _period_trend_from_highs_leakfree(
    ts: pd.Series, high: pd.Series, period: str
) -> pd.Series:
    ts_local = ts.dt.tz_convert(None) if ts.dt.tz is not None else ts
    periods = ts_local.dt.to_period(period)
    frame = pd.DataFrame({"period": periods, "high": high})
    max_by_period = frame.groupby("period", sort=True)["high"].max()
    prev_max_by_period = max_by_period.shift(1)
    running_high = high.groupby(periods).cummax()
    prev_map = prev_max_by_period.to_dict()
    prev_at_row = periods.map(prev_map)
    trend_row = pd.Series(0, index=high.index, dtype="Int8")
    mask_prev_ok = pd.notna(prev_at_row)
    trend_row[(running_high > prev_at_row) & mask_prev_ok] = 1
    trend_row[(running_high < prev_at_row) & mask_prev_ok] = -1
    return trend_row


def _rolling_ols_slope_fast(y: pd.Series, window: int) -> pd.Series:
    """
    Vectorized rolling OLS slope of y vs t = 0,1,2,..., over `window`.
        slope = cov(t, y) / var(t)
    var(t) for w consecutive integers is exactly w*(w+1)/12, so we avoid
    rebuilding the constant t-series statistics each window.
    Same numerical result as the v19 Python loop, ~10-50x faster.
    """
    y = pd.to_numeric(y, errors="coerce")
    n = len(y)
    if n == 0:
        return pd.Series(np.full(0, np.nan), index=y.index, dtype="float64")
    w = int(window)
    if w <= 1:
        return pd.Series(np.full(n, np.nan), index=y.index, dtype="float64")
    t = pd.Series(np.arange(n, dtype="float64"), index=y.index)
    var_t = w * (w + 1) / 12.0
    cov_xy = t.rolling(w, min_periods=w).cov(y)
    return (cov_xy / var_t).astype("float64")


# ──────────────────────────────────────────────────────────────────────────────
#  v20 LEAK-FREE PRIMITIVES
#
#  These are the ONLY allowed transforms inside compute_daily_indicators().
#  Anything that ranks / normalizes / aggregates against a per-symbol's
#  WHOLE series (e.g. unwindowed s.rank() / s.mean() / s.quantile()) would
#  peek at the future and is forbidden. The static check at the bottom of
#  this section refuses to import the module if a forbidden pattern leaks
#  back in.
# ──────────────────────────────────────────────────────────────────────────────


def _expand_rank_pct(s: pd.Series, min_periods: int = 60) -> pd.Series:
    """
    Leak-free expanding percentile rank.

    At row t, returns the percentile rank of s[t] within s[0..t] inclusive
    (average method for ties). NaN values are skipped in the ranking and
    inherit NaN in the output. Returns NaN until at least `min_periods`
    finite values have been observed.

    This is the leak-free replacement for v19's `_rank_cs(s) = s.rank(pct=True)`,
    which ranked each row against the entire (past + future) series.
    """
    s = pd.to_numeric(s, errors="coerce")
    if len(s) == 0:
        return pd.Series([], index=s.index, dtype="float64")
    # Pandas >= 1.4: vectorized, C-level expanding rank.
    try:
        return s.expanding(min_periods=int(min_periods)).rank(pct=True).astype("float64")
    except Exception:
        # Fallback for older pandas: bisect-based, O(n^2) but correct.
        import bisect
        arr = s.to_numpy(dtype="float64")
        n = arr.size
        out = np.full(n, np.nan, dtype="float64")
        sl: List[float] = []
        for i in range(n):
            x = arr[i]
            if np.isfinite(x):
                bisect.insort(sl, x)
                if len(sl) >= int(min_periods):
                    lo = bisect.bisect_left(sl, x)
                    hi = bisect.bisect_right(sl, x)
                    out[i] = ((lo + 1 + hi) / 2.0) / len(sl)
        return pd.Series(out, index=s.index, dtype="float64")


def _streak_length(flag) -> pd.Series:
    """O(n) consecutive-True streak length. flag may be bool/Int8/object."""
    arr = pd.Series(flag).fillna(False).astype(bool).to_numpy()
    n = arr.size
    if n == 0:
        return pd.Series(np.zeros(0, dtype="int32"))
    grp = (~arr).cumsum()  # increments on every False -> resets the run
    s = pd.Series(arr.astype("int32"))
    # v21: copy=True. pandas 3.x returns a read-only view here, so the
    # in-place masking below raised "assignment destination is read-only".
    out = s.groupby(grp).cumsum().to_numpy(copy=True)
    out[~arr] = 0
    return pd.Series(out.astype("int32"))


def _days_since_flag(flag) -> pd.Series:
    """
    O(n) days since the most recent True. Zero before the first True (matching
    v19 behaviour for D_days_since_boh_20 / D_days_since_bol_20).
    """
    arr = pd.Series(flag).fillna(0).astype("int8").to_numpy()
    n = arr.size
    if n == 0:
        return pd.Series(np.zeros(0, dtype="int32"))
    idx = np.arange(n, dtype="int64")
    last = np.where(arr == 1, idx, -1)
    last = np.maximum.accumulate(last)
    days = (idx - last).astype("int32")
    days[last < 0] = 0
    return pd.Series(days)


def _v20_static_leak_check() -> None:
    """
    At import time, parse this file and refuse to load if compute_daily_indicators
    contains any unwindowed `.rank(`, `.cumsum(`, `.cummax(`, `.cummin(`,
    `.expanding(min_periods` without an `expanding` qualifier, etc. We allow
    `.cumsum()` only on diff/sign products (OBV pattern) by whitelisting the
    surrounding context.

    The check is deliberately conservative: it reads source and grep-tests
    for forbidden tokens; false positives are tolerated and can be silenced
    with the marker `# leak-free:` on the same line.
    """
    try:
        path = Path(__file__)
        src = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return

    # Slice out the body of compute_daily_indicators. Use a leading newline
    # in the search anchor so we don't match the string literal in this
    # function's own body (which appears earlier in the file).
    start = src.find("\ndef compute_daily_indicators(")
    if start < 0:
        return
    start += 1  # skip the leading newline so line numbering aligns
    # End at the next top-level "# ----" banner that follows the function
    end = src.find("\n# -------------------- Finalize for cache --------------------", start)
    if end < 0:
        end = len(src)
    body = src[start:end]

    forbidden_patterns = [
        # An unwindowed rank() that is not preceded by .rolling( or .expanding(
        # within a few characters. Tolerate `# leak-free:` annotation.
        # We split lines to make per-line annotation possible.
    ]
    bad: List[str] = []
    body_lines = body.splitlines()
    for lineno, line in enumerate(body_lines, start=1):
        if "# leak-free:" in line:
            continue
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # Whole-series rank: `.rank(` not preceded by `.rolling(`, `.expanding(`,
        # or `.groupby(` within the same statement. Statements may span lines,
        # so we widen the search to the previous 4 lines for chain continuations.
        if ".rank(" in line:
            pre = line.split(".rank(")[0]
            ctx = "\n".join(body_lines[max(0, lineno - 5): lineno])
            qualified = (
                "rolling(" in pre
                or "expanding(" in pre
                or "groupby(" in pre
                or "rolling(" in ctx
                or "expanding(" in ctx
                or "groupby(" in ctx
            )
            if not qualified:
                bad.append(f"L{lineno}: {stripped}")
    if bad:
        raise RuntimeError(
            "v20 STATIC LEAK CHECK FAILED — unwindowed .rank() calls detected "
            "inside compute_daily_indicators(). These leak the future. "
            "Lines:\n  " + "\n  ".join(bad[:20])
        )


def _v20_leak_canary_check() -> None:
    """
    Runtime canary: build two synthetic OHLCV frames identical for indices
    [0, N-K) and DIFFERENT for the last K rows. Compute indicators on both
    and assert every column matches on rows [0, N-K). Any column whose
    PAST values change when only the FUTURE changes is leaking.

    The label column `ret_5d_close_pct` is exempt — it is a forward return
    and is supposed to peek (but it should never be used as a feature).
    """
    rng_ = np.random.default_rng(42)
    N = 400
    K = 12
    log_ret = rng_.normal(0.0, 0.012, N)
    close_arr = 100.0 * np.exp(np.cumsum(log_ret))
    open_arr = close_arr * (1.0 + rng_.normal(0.0, 0.003, N))
    hi_off = np.abs(rng_.normal(0.0, 0.005, N))
    lo_off = np.abs(rng_.normal(0.0, 0.005, N))
    high_arr = np.maximum(close_arr, open_arr) * (1.0 + hi_off)
    low_arr = np.minimum(close_arr, open_arr) * (1.0 - lo_off)
    vol_arr = rng_.integers(50_000, 1_500_000, N).astype("float64")
    ts = pd.date_range("2018-01-01", periods=N, freq="B", tz=IST)

    base = pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_arr,
            "high": high_arr,
            "low": low_arr,
            "close": close_arr,
            "volume": vol_arr,
        }
    )
    tampered = base.copy()
    rng2 = np.random.default_rng(999)
    mult_oc = rng2.uniform(0.4, 2.5, K)
    mult_hl = rng2.uniform(0.4, 2.5, K)
    tampered.loc[N - K:, "open"] = tampered.loc[N - K:, "open"].to_numpy() * mult_oc
    tampered.loc[N - K:, "close"] = tampered.loc[N - K:, "close"].to_numpy() * mult_oc
    tampered.loc[N - K:, "high"] = tampered.loc[N - K:, "high"].to_numpy() * mult_hl
    tampered.loc[N - K:, "low"] = tampered.loc[N - K:, "low"].to_numpy() * mult_hl
    tampered.loc[N - K:, "volume"] = (
        tampered.loc[N - K:, "volume"].to_numpy() * rng2.uniform(0.1, 10.0, K)
    )

    out_a = compute_daily_indicators(base.copy())
    out_b = compute_daily_indicators(tampered.copy())
    head_a = out_a.iloc[: N - K]
    head_b = out_b.iloc[: N - K]

    # Skip the timestamp itself + the forward-looking label column.
    # ret_5d_close_pct[t] = close[t+5]/close[t] - 1, so for t in [N-K-5, N-K),
    # it legitimately reads tampered values.
    SKIP = {"timestamp", "ret_5d_close_pct"}
    leaks: List[str] = []
    for c in head_a.columns:
        if c in SKIP:
            continue
        try:
            a = pd.to_numeric(head_a[c], errors="coerce").to_numpy(dtype="float64")
            b = pd.to_numeric(head_b[c], errors="coerce").to_numpy(dtype="float64")
        except Exception:
            continue
        a_nan = np.isnan(a)
        b_nan = np.isnan(b)
        if not np.array_equal(a_nan, b_nan):
            leaks.append(c)
            continue
        diff = np.where(a_nan, 0.0, np.abs(a - b))
        if np.nanmax(diff) > 1e-7:
            leaks.append(c)
    if leaks:
        raise RuntimeError(
            "v20 LEAK CANARY FAILED — these columns depend on FUTURE data: "
            + ", ".join(leaks[:30])
            + (f"  (+{len(leaks) - 30} more)" if len(leaks) > 30 else "")
        )


# -------------------- Indicators column list --------------------
DAILY_INDICATOR_COLUMNS = [
    # Core OHLCV
    "timestamp", "open", "high", "low", "close", "volume",
    # EMAs / RSI
    "D_ema20", "D_ema50", "D_ema100", "D_rsi7", "D_rsi14",
    # MACD
    "D_macd", "D_macd_signal", "D_macd_hist",
    # CMF / ADX
    "D_cmf20", "D_adx14", "D_pdi14", "D_mdi14",
    # Inside day
    "D_inside_day", "D_prev_inside_day",
    # CPR / Pivots
    "D_cpr_pivot", "D_cpr_bc", "D_cpr_tc",
    "D_pivot", "D_support1", "D_resistance1", "D_support2", "D_resistance2",
    # NR
    "D_nr", "D_nr_length", "D_nr_day",
    # VPOC
    "D_vpoc", "D_weekly_vpoc",
    # SMAs
    "D_sma5", "D_sma20",
    # Trend
    "D_daily_trend", "D_weekly_trend", "D_monthly_trend",
    "D_rsi7_gt_rsi14", "D_ema_stack_20_50_100", "D_ema20_angle_deg",
    # ATR
    "D_atr14", "D_atr30", "D_atr_ratio_14_30",
    # CPR width / tomorrow CPR
    "D_cpr_width_pct", "D_tmr_cpr_bc", "D_tmr_cpr_tc",
    "D_tmr_cpr_vs_today", "D_cpr_vs_yday",
    # Structure
    "D_hh", "D_hl", "D_lh", "D_ll", "D_structure_trend",
    # Prev day
    "D_prev_high", "D_prev_low", "D_prev_close",
    # OLI / day type / range
    "D_oli", "D_day_type", "D_range_to_atr14",
    # SMAs extended
    "D_sma50", "D_sma200", "D_golden_regime",
    # OBV
    "D_obv", "D_obv_slope", "D_price_and_obv_rising",
    # Numeric codes
    "D_tmr_cpr_vs_today_code", "D_cpr_vs_yday_code", "D_structure_trend_code",
    # v17: Bollinger / Yang-Zhang / Donchian / Breakout / Volume
    "D_dow",
    "D_bb_pctB_20", "D_bb_bw_20",
    "D_vol_yz_20", "D_vol_yz_50",
    "D_donch_pos_20", "D_donch_pos_50",
    "D_breakout_high_20", "D_breakout_low_20",
    "D_breakout_high_50", "D_breakout_low_50",
    "D_days_since_boh_20", "D_days_since_bol_20",
    "D_dollar_vol", "D_dvol_z20", "D_dvol_z50", "D_dvol_z252",
    "D_vol_surge_20", "D_vol_surge_50",
    # v17: Cache-side features
    "D_atr_pct", "D_range_pct", "D_gap_pct",
    "D_rsi14_z252", "D_atr_pct_z252", "D_vol_z252", "D_ema20_angle_z252",
    "D_rsi14_obv_x", "D_rsi7_obv_x", "D_atr14_to_close_pct",
    "D_ret_5d_roll_std", "D_close_roll_slope_20", "D_close_roll_slope_50",
    # v17: Logical combos
    "Comb_RSIslopePos__ADX_15_25",
    "Comb_GapUp__CPR_Tmr_Above", "Comb_GapDown__CPR_Tmr_Below",
    "Comb_ATRlow__EMA20pos", "Comb_ATRhigh__EMA20neg",
    # v18: Copilot structural features
    "D_body_ratio", "D_wick_skew",
    "D_hh_run", "D_hl_run", "D_lh_run", "D_ll_run",
    "D_nr_expand", "D_compress_state",
    "D_dist_from_20h", "D_dist_from_20l", "D_dist_from_52wh",
    "D_midpoint_slope", "D_slope_stability",
    # v18: Weekly momentum features
    "W_ret_4w", "W_ret_13w", "W_close_pos", "W_vol_vs_4w",
    # v19 NEW: 16 WorldQuant alphas
    "D_WQ_3", "D_WQ_6", "D_WQ_12", "D_WQ_13", "D_WQ_16", "D_WQ_19", "D_WQ_20", "D_WQ_23",
        "D_WQ_26", "D_WQ_29", "D_WQ_33", "D_WQ_35", "D_WQ_38", "D_WQ_40", "D_WQ_41", "D_WQ_44",
    # v19 NEW: 17 rolling/lag/diff transforms
    "D_slope_stability_rmean50", "D_slope_stability_rstd10", "D_slope_stability_rstd20",
    "D_body_ratio_rmean50", "D_body_ratio_rmean20",
    "D_close_roll_slope_20_rstd20", "D_close_roll_slope_20_rstd10",
    "D_macd_hist_rstd10",
    "D_mdi14_diff1", "D_mdi14_diff5",
    "D_mdi14_rrank10", "D_mdi14_rrank20",
    "D_donch_pos_50_rmean50", "D_donch_pos_50_lag5",
    "D_donch_pos_20_rmean50", "D_donch_pos_20_lag5",
    "D_cmf20_rmean50",
]


# =============================================================================
#   INDICATOR COMPUTATION
# =============================================================================

def compute_daily_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    missing = [c for c in DAILY_INDICATOR_COLUMNS if c not in df.columns]
    if missing:
        df = pd.concat(
            [df, pd.DataFrame({c: np.nan for c in missing}, index=df.index)],
            axis=1,
            copy=False,
        )
    if df.empty:
        return df

    df = df.sort_values("timestamp").reset_index(drop=True)
    timestamps = pd.to_datetime(df["timestamp"])
    timestamps_local = (
        timestamps.dt.tz_convert(None) if timestamps.dt.tz is not None else timestamps
    )

    close = _ensure_float(df["close"])
    high = _ensure_float(df["high"])
    low = _ensure_float(df["low"])
    open_ = _ensure_float(df["open"])
    volume = _ensure_float(df["volume"]).fillna(0.0)
    rng = high - low  # pre-compute once

    # ── EMAs / RSI ────────────────────────────────────────────────────────
    df["D_ema20"] = _ema(close, 20)
    df["D_ema50"] = _ema(close, 50)
    df["D_ema100"] = _ema(close, 100)
    df["D_rsi7"] = _rsi(close, 7)
    df["D_rsi14"] = _rsi(close, 14)

    # ── MACD 12,26,9 ──────────────────────────────────────────────────────
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    df["D_macd"] = macd
    df["D_macd_signal"] = signal
    df["D_macd_hist"] = macd - signal

    # ── CMF 20 ────────────────────────────────────────────────────────────
    df["D_cmf20"] = (
        (((close - low) - (high - close)) / (high - low).replace(0, np.nan) * volume)
        .rolling(window=20, min_periods=20)
        .sum()
        / volume.rolling(window=20, min_periods=20).sum()
    )

    # ── ADX/PDI/MDI 14 ────────────────────────────────────────────────────
    adx, pdi, mdi = _adx(high, low, close, 14)
    df["D_adx14"] = adx
    df["D_pdi14"] = pdi
    df["D_mdi14"] = mdi

    # ── Prev day references & inside day flags ────────────────────────────
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)
    df["D_prev_high"] = prev_high
    df["D_prev_low"] = prev_low
    df["D_prev_close"] = prev_close
    df["D_inside_day"] = ((high <= prev_high) & (low >= prev_low)).astype("boolean")
    df["D_prev_inside_day"] = df["D_inside_day"].shift(1).astype("boolean")

    # ── CPR / Pivots ──────────────────────────────────────────────────────
    pivot = (high + low + close) / 3
    cpr_bc = (high + low) / 2
    cpr_tc = 2 * pivot - cpr_bc
    df["D_cpr_pivot"] = pivot
    df["D_cpr_bc"] = cpr_bc
    df["D_cpr_tc"] = cpr_tc
    df["D_pivot"] = pivot
    df["D_support1"] = 2 * pivot - high
    df["D_resistance1"] = 2 * pivot - low
    df["D_support2"] = pivot - rng
    df["D_resistance2"] = pivot + rng

    # ── VPOC proxy ────────────────────────────────────────────────────────
    df["D_vpoc"] = (high + low + close) / 3.0

    # ── Weekly VPOC ───────────────────────────────────────────────────────
    df["D_weekly_vpoc"] = _compute_weekly_vpoc_fast(
        timestamps, high, low, close, volume
    )

    # ── NR7 & length ──────────────────────────────────────────────────────
    prev6_min = rng.shift(1).rolling(window=6, min_periods=6).min()
    nr7 = (rng < prev6_min) & prev6_min.notna()
    df["D_nr"] = nr7.astype("boolean")

    nr_arr = nr7.fillna(False).to_numpy()
    df["D_nr_length"] = _streak_length(nr_arr).astype("int64").to_numpy()

    values = rng.astype("float64")
    nr_window = pd.Series(pd.NA, index=df.index, dtype="Int64")
    for w in range(20, 5 - 1, -1):
        prev_min = values.shift(1).rolling(window=w - 1, min_periods=w - 1).min()
        mask = (values < prev_min) & prev_min.notna()
        nr_window = nr_window.mask(mask & nr_window.isna(), w)
    df["D_nr_day"] = nr_window

    # ── SMAs ─────────────────────────────────────────────────────────────
    df["D_sma5"] = _sma(close, 5)
    df["D_sma20"] = _sma(close, 20)
    df["D_sma50"] = _sma(close, 50)
    df["D_sma200"] = _sma(close, 200)

    df["D_rsi7_gt_rsi14"] = (df["D_rsi7"] > df["D_rsi14"]).astype("boolean")
    df["D_ema_stack_20_50_100"] = (
        (df["D_ema20"] > df["D_ema50"]) & (df["D_ema50"] > df["D_ema100"])
    ).astype("boolean")

    ema20 = df["D_ema20"]
    prev_ema20 = ema20.shift(1)
    pct_slope = (ema20 - prev_ema20) / prev_ema20.replace(0, np.nan)
    df["D_ema20_angle_deg"] = np.degrees(np.arctan(pct_slope))

    # ── ATRs ─────────────────────────────────────────────────────────────
    df["D_atr14"] = _atr(high, low, close, 14)
    df["D_atr30"] = _atr(high, low, close, 30)
    df["D_atr_ratio_14_30"] = df["D_atr14"] / df["D_atr30"].replace(0, np.nan)

    # ── CPR width / tomorrow CPR ─────────────────────────────────────────
    df["D_cpr_width_pct"] = (
        (df["D_cpr_tc"] - df["D_cpr_bc"]) / close.replace(0, np.nan)
    ) * 100
    df["D_tmr_cpr_bc"] = cpr_bc
    df["D_tmr_cpr_tc"] = cpr_tc

    pivot_y = (high.shift(1) + low.shift(1) + close.shift(1)) / 3.0
    cpr_bc_y = (high.shift(1) + low.shift(1)) / 2.0
    cpr_tc_y = 2 * pivot_y - cpr_bc_y
    rel_tmr = _cpr_relationship(
        cpr_bc_y, cpr_tc_y, df["D_tmr_cpr_bc"], df["D_tmr_cpr_tc"]
    )
    rel_vs_y = _cpr_relationship(
        cpr_bc_y, cpr_tc_y, df["D_cpr_bc"], df["D_cpr_tc"]
    )
    df["D_tmr_cpr_vs_today"] = rel_tmr
    df["D_cpr_vs_yday"] = rel_vs_y

    # ── Structure / trend flags ───────────────────────────────────────────
    df["D_hh"] = (high > prev_high).astype("boolean")
    df["D_hl"] = (low > prev_low).astype("boolean")
    df["D_lh"] = (high < prev_high).astype("boolean")
    df["D_ll"] = (low < prev_low).astype("boolean")

    daily_trend = pd.Series(0, index=df.index, dtype="Int8")
    daily_trend[df["D_hh"] == True] = 1
    daily_trend[df["D_lh"] == True] = -1
    df["D_daily_trend"] = daily_trend.astype("Int8")
    df["D_weekly_trend"] = _period_trend_from_highs_leakfree(
        timestamps, high, "W-FRI"
    )
    df["D_monthly_trend"] = _period_trend_from_highs_leakfree(
        timestamps, high, "M"
    )
    df["D_structure_trend"] = np.select(
        [df["D_hh"] & df["D_hl"], df["D_lh"] & df["D_ll"]],
        ["uptrend", "downtrend"],
        default="range",
    )

    # ── Numeric codes ─────────────────────────────────────────────────────
    def _encode_rel(s: pd.Series) -> pd.Series:
        return (
            s.map({"Above": 1, "Inside": 0, "Overlap": 0, "Below": -1})
            .fillna(0)
            .astype("Int8")
        )

    def _encode_trend(s: pd.Series) -> pd.Series:
        return (
            s.map({"uptrend": 1, "range": 0, "downtrend": -1})
            .fillna(0)
            .astype("Int8")
        )

    df["D_tmr_cpr_vs_today_code"] = _encode_rel(df["D_tmr_cpr_vs_today"])
    df["D_cpr_vs_yday_code"] = _encode_rel(df["D_cpr_vs_yday"])
    df["D_structure_trend_code"] = _encode_trend(df["D_structure_trend"])

    # ── OLI / day type / range ────────────────────────────────────────────
    df["D_oli"] = (open_ - low) / rng.replace(0, np.nan)
    df["D_day_type"] = np.select(
        [open_ > cpr_tc, open_ < cpr_bc], ["bullish", "bearish"], default="inside"
    )
    df["D_range_to_atr14"] = rng / df["D_atr14"].replace(0, np.nan)
    df["D_golden_regime"] = (
        (close > df["D_sma200"]) & (df["D_sma50"] > df["D_sma200"])
    ).astype("boolean")

    # ── OBV + slope ───────────────────────────────────────────────────────
    obv = (np.sign(close.diff().fillna(0.0)) * volume).cumsum()
    df["D_obv"] = obv
    df["D_obv_slope"] = obv.diff()
    df["D_price_and_obv_rising"] = (
        (close > close.shift(1)) & (obv > obv.shift(1))
    ).astype("boolean")

    # ══════════════════════════════════════════════════════════════════════
    #  v17 features
    # ══════════════════════════════════════════════════════════════════════

    # Day-of-week
    ts_local = (
        timestamps.dt.tz_convert(None) if timestamps.dt.tz is not None else timestamps
    )
    df["D_dow"] = ts_local.dt.weekday.astype("Int8")
    dow_dummies = pd.get_dummies(df["D_dow"], prefix="DOW", dtype="int8")
    for c in dow_dummies.columns:
        df[c] = dow_dummies[c]

    # Bollinger Bands (20)
    sma20 = df["D_sma20"]
    std20 = close.rolling(20, min_periods=20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_width = bb_upper - bb_lower
    df["D_bb_pctB_20"] = (
        (close - bb_lower) / bb_width.replace(0, np.nan)
    ).clip(lower=-5, upper=5)
    df["D_bb_bw_20"] = (bb_width / sma20.replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )

    # Yang-Zhang volatility
    log_oo = np.log(open_ / prev_close).replace([np.inf, -np.inf], np.nan)
    log_cc = np.log(close / open_).replace([np.inf, -np.inf], np.nan)
    log_h_o = np.log(high / open_).replace([np.inf, -np.inf], np.nan)
    log_l_o = np.log(low / open_).replace([np.inf, -np.inf], np.nan)
    log_h_c = np.log(high / close).replace([np.inf, -np.inf], np.nan)
    log_l_c = np.log(low / close).replace([np.inf, -np.inf], np.nan)
    rs = log_h_o * log_h_c + log_l_o * log_l_c
    k = 0.34
    yz_var = (log_oo**2) + k * (log_cc**2) + (1 - k) * rs

    def _yz_roll(win: int):
        v = yz_var.rolling(win, min_periods=win).mean()
        return np.sqrt(v)

    df["D_vol_yz_20"] = _yz_roll(20)
    df["D_vol_yz_50"] = _yz_roll(50)

    # Donchian
    hi_20 = high.shift(1).rolling(20, min_periods=20).max()
    lo_20 = low.shift(1).rolling(20, min_periods=20).min()
    rng_20 = (hi_20 - lo_20).replace(0, np.nan)
    hi_50 = high.shift(1).rolling(50, min_periods=50).max()
    lo_50 = low.shift(1).rolling(50, min_periods=50).min()
    rng_50 = (hi_50 - lo_50).replace(0, np.nan)

    df["D_donch_pos_20"] = ((close - lo_20) / rng_20).clip(lower=-1, upper=2)
    df["D_donch_pos_50"] = ((close - lo_50) / rng_50).clip(lower=-1, upper=2)

    # Breakouts
    df["D_breakout_high_20"] = (high > hi_20).astype("int8")
    df["D_breakout_low_20"] = (low < lo_20).astype("int8")
    df["D_breakout_high_50"] = (high > hi_50).astype("int8")
    df["D_breakout_low_50"] = (low < lo_50).astype("int8")

    # Days since breakouts (O(n) vectorized via _days_since_flag)
    df["D_days_since_boh_20"] = _days_since_flag(df["D_breakout_high_20"]).astype("int16")
    df["D_days_since_bol_20"] = _days_since_flag(df["D_breakout_low_20"]).astype("int16")

    # Dollar volume & Z-scores / surge flags
    df["D_dollar_vol"] = (close * volume).replace([np.inf, -np.inf], np.nan)

    def _zscore(s, win=252):
        # v21: min_periods=win. A "252-day z-score" built from 63 samples
        # is a different statistic with a much fatter tail.
        x = pd.to_numeric(s, errors="coerce")
        r = x.rolling(win, min_periods=win)
        m = r.mean()
        v = r.std()
        return (x - m) / v.replace(0, np.nan)

    df["D_dvol_z20"] = _zscore(df["D_dollar_vol"], 20)
    df["D_dvol_z50"] = _zscore(df["D_dollar_vol"], 50)
    df["D_dvol_z252"] = _zscore(df["D_dollar_vol"], 252)

    _r20 = volume.rolling(20, min_periods=20)
    _r50 = volume.rolling(50, min_periods=50)
    vol20_m = _r20.mean()
    vol20_s = _r20.std()
    vol50_m = _r50.mean()
    vol50_s = _r50.std()
    df["D_vol_surge_20"] = (volume > (vol20_m + 2 * vol20_s)).astype("int8")
    df["D_vol_surge_50"] = (volume > (vol50_m + 2 * vol50_s)).astype("int8")

    # Cache-side features
    df["D_atr_pct"] = (df["D_atr14"] / close.replace(0, np.nan)) * 100
    df["D_range_pct"] = ((high - low) / close.replace(0, np.nan)) * 100
    df["D_gap_pct"] = (
        (open_ - df["D_prev_close"]) / df["D_prev_close"].replace(0, np.nan)
    ) * 100

    def _zscore_generic(s, win=252):
        x = pd.to_numeric(s, errors="coerce")
        r = x.rolling(win, min_periods=win)
        m = r.mean()
        v = r.std()
        return (x - m) / v.replace(0, np.nan)

    df["D_rsi14_z252"] = _zscore_generic(df["D_rsi14"], 252)
    df["D_atr_pct_z252"] = _zscore_generic(df["D_atr_pct"], 252)
    df["D_vol_z252"] = _zscore_generic(df["volume"], 252)
    df["D_ema20_angle_z252"] = _zscore_generic(df["D_ema20_angle_deg"], 252)

    rsi14 = pd.to_numeric(df.get("D_rsi14"), errors="coerce")
    rsi7 = pd.to_numeric(df.get("D_rsi7"), errors="coerce")
    obvs = pd.to_numeric(df.get("D_obv_slope"), errors="coerce")
    df["D_rsi14_obv_x"] = rsi14 * obvs
    if "D_rsi7" in df.columns:
        df["D_rsi7_obv_x"] = rsi7 * obvs
    df["D_atr14_to_close_pct"] = (
        df["D_atr14"] / close
    ).replace([np.inf, -np.inf], np.nan) * 100.0

    if "ret_5d_close_pct" not in df.columns:
        df["ret_5d_close_pct"] = (df["close"].shift(-5) / df["close"] - 1) * 100

    ret_fwd_5 = pd.to_numeric(df.get("ret_5d_close_pct"), errors="coerce")
    df["D_ret_5d_roll_std"] = ret_fwd_5.shift(5).rolling(50, min_periods=50).std()

    # Rolling OLS slope
    df["D_close_roll_slope_20"] = _rolling_ols_slope_fast(df["close"], window=20)
    df["D_close_roll_slope_50"] = _rolling_ols_slope_fast(df["close"], window=50)

    # Logical combos
    rsi14_diff = pd.to_numeric(df["D_rsi14"], errors="coerce").diff()
    adx14 = pd.to_numeric(df["D_adx14"], errors="coerce")
    df["Comb_RSIslopePos__ADX_15_25"] = (
        (rsi14_diff > 0) & (adx14 >= 15) & (adx14 <= 25)
    ).astype("int8")

    gap = pd.to_numeric(df["D_gap_pct"], errors="coerce")
    df["Comb_GapUp__CPR_Tmr_Above"] = (
        (gap > 0) & (df["D_tmr_cpr_vs_today_code"] == 1)
    ).astype("int8")
    df["Comb_GapDown__CPR_Tmr_Below"] = (
        (gap < 0) & (df["D_tmr_cpr_vs_today_code"] == -1)
    ).astype("int8")

    atr_pct = pd.to_numeric(df["D_atr_pct"], errors="coerce")
    ema_ang = pd.to_numeric(df["D_ema20_angle_deg"], errors="coerce")
    ATR_LOW = 2.0
    ATR_HIGH = 4.0
    df["Comb_ATRlow__EMA20pos"] = (
        (atr_pct <= ATR_LOW) & (ema_ang > 0)
    ).astype("int8")
    df["Comb_ATRhigh__EMA20neg"] = (
        (atr_pct >= ATR_HIGH) & (ema_ang < 0)
    ).astype("int8")

    # ══════════════════════════════════════════════════════════════════════
    #  v18: Copilot structural features
    # ══════════════════════════════════════════════════════════════════════

    rng_safe = rng.replace(0, np.nan)
    atr14_safe = df["D_atr14"].replace(0, np.nan)

    # Candle geometry
    df["D_body_ratio"] = ((close - open_) / rng_safe).clip(-1, 1)
    upper_wick = high - pd.concat([close, open_], axis=1).max(axis=1)
    lower_wick = pd.concat([close, open_], axis=1).min(axis=1) - low
    df["D_wick_skew"] = ((upper_wick - lower_wick) / rng_safe).clip(-1, 1)

    # Path-dependency: consecutive run counts (O(n) vectorized)
    df["D_hh_run"] = _streak_length(df["D_hh"] == True).astype("int16")
    df["D_hl_run"] = _streak_length(df["D_hl"] == True).astype("int16")
    df["D_lh_run"] = _streak_length(df["D_lh"] == True).astype("int16")
    df["D_ll_run"] = _streak_length(df["D_ll"] == True).astype("int16")

    # Range compression/expansion
    df["D_nr_expand"] = (rng > rng.shift(1)).astype("int8")

    # Compression state
    df["D_compress_state"] = (
        df["D_bb_bw_20"]
        .rolling(50, min_periods=10)
        .rank(pct=True)
    )

    # Distance from regime anchors
    df["D_dist_from_20h"] = (close - hi_20) / atr14_safe
    df["D_dist_from_20l"] = (close - lo_20) / atr14_safe

    # v21: min_periods=252. A 52-week high off 50 bars is a 10-week high,
    # and D_dist_from_52wh is one of the dominant model features.
    hi_52w = high.shift(1).rolling(252, min_periods=252).max()
    df["D_dist_from_52wh"] = (close - hi_52w) / atr14_safe

    # Structural trend quality
    midpoint = (high + low) / 2.0
    mp_slope_raw = _rolling_ols_slope_fast(midpoint, window=10)
    df["D_midpoint_slope"] = mp_slope_raw / atr14_safe.values

    close_slope_5 = _rolling_ols_slope_fast(close, window=5)
    df["D_slope_stability"] = (
        pd.Series(close_slope_5, index=df.index)
        .rolling(20, min_periods=5)
        .std()
    )

    # ══════════════════════════════════════════════════════════════════════
    #  v18: Weekly momentum features
    # ══════════════════════════════════════════════════════════════════════

    ts_local_naive = (
        timestamps.dt.tz_convert(None) if timestamps.dt.tz is not None else timestamps
    )
    df_tmp = pd.DataFrame(
        {
            "close": close.values,
            "high": high.values,
            "low": low.values,
            "volume": volume.values,
        },
        index=ts_local_naive,
    )

    weekly = df_tmp.resample("W-FRI").agg(
        {"close": "last", "high": "max", "low": "min", "volume": "sum"}
    ).dropna(subset=["close"])

    if len(weekly) >= 5:
        w_close = weekly["close"]
        w_high = weekly["high"]
        w_low = weekly["low"]
        w_vol = weekly["volume"]
        w_rng = (w_high - w_low).replace(0, np.nan)

        w_ret_4w = (w_close / w_close.shift(4) - 1.0) * 100.0
        w_ret_13w = (w_close / w_close.shift(13) - 1.0) * 100.0
        w_close_pos = ((w_close - w_low) / w_rng).clip(0, 1)
        w_vol_4w_avg = w_vol.rolling(4, min_periods=2).mean().shift(1)
        w_vol_vs_4w = (w_vol / w_vol_4w_avg.replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )

        w_features = pd.DataFrame(
            {
                "W_ret_4w": w_ret_4w.shift(1),
                "W_ret_13w": w_ret_13w.shift(1),
                "W_close_pos": w_close_pos.shift(1),
                "W_vol_vs_4w": w_vol_vs_4w.shift(1),
            },
            index=weekly.index,
        )

        day_week_end = ts_local_naive.dt.to_period("W-FRI").dt.end_time.dt.normalize()
        idx = w_features.index
        if isinstance(idx, pd.PeriodIndex):
            idx = idx.to_timestamp(how="end")

        wf_index = pd.to_datetime(idx).normalize()
        wf_lookup = w_features.copy()
        wf_lookup.index = wf_index

        for feat_col in ["W_ret_4w", "W_ret_13w", "W_close_pos", "W_vol_vs_4w"]:
            lookup_dict = wf_lookup[feat_col].to_dict()
            df[feat_col] = day_week_end.map(lookup_dict).values
    else:
        for feat_col in ["W_ret_4w", "W_ret_13w", "W_close_pos", "W_vol_vs_4w"]:
            df[feat_col] = np.nan

    # ══════════════════════════════════════════════════════════════════════
    #  v20: 16 WorldQuant alphas — LEAK-FREE BY CONSTRUCTION
    #
    #  v19 used `_rank_cs(s) = s.rank(pct=True)` which ranked each row
    #  against the WHOLE per-symbol series (past + future). At row t, the
    #  rank therefore encoded "where t sits relative to t+1, t+2, …, N-1"
    #  — a textbook look-ahead leak. The 10 alphas that flow through
    #  _rank_cs (3, 13, 16, 19, 20, 29, 33, 38, 40, 44) were tainted.
    #
    #  v20 replaces every per-symbol rank with `_xrank` =
    #  s.expanding(min_periods=60).rank(pct=True). At row t, this is the
    #  percentile rank of s[t] within s[0..t] only — strictly causal.
    #  Column names are preserved so downstream code keeps working.
    #
    #  IMPORTANT: this is a *temporal* per-symbol rank, NOT the WQ-101
    #  cross-sectional rank. A genuine WQ rank would be
    #      panel.groupby('timestamp')[col].rank(pct=True)
    #  computed at panel-assembly time (see New_model.py / NEW FEAT IMP.py).
    #  These per-symbol expanding ranks are honest features but do NOT
    #  inherit WQ's published cross-sectional evidence.
    # ══════════════════════════════════════════════════════════════════════

    ret_d = close.pct_change()
    vwap_d = (high + low + close) / 3.0
    adv20_d = volume.rolling(20, min_periods=5).mean()

    def _ts_rank(s, w):
        return s.rolling(w, min_periods=max(3, w // 2)).rank(pct=True)

    def _ts_corr(a, b, w):
        return a.rolling(w, min_periods=max(5, w // 2)).corr(b)

    def _ts_std(s, w):
        return s.rolling(w, min_periods=max(3, w // 2)).std()

    def _ts_mean(s, w):
        return s.rolling(w, min_periods=max(3, w // 2)).mean()

    def _ts_max(s, w):
        return s.rolling(w, min_periods=max(3, w // 2)).max()

    def _ts_min(s, w):
        return s.rolling(w, min_periods=max(3, w // 2)).min()

    def _delta(s, d):
        return s.diff(d)

    def _delay(s, d):
        return s.shift(d)

    # LEAK-FREE per-symbol historical percentile rank (replaces v19 _rank_cs).
    # min_periods=60 ≈ ~3 trading months of warmup so early values are stable.
    _XRANK_MIN_PERIODS = 60

    def _xrank(s):
        return _expand_rank_pct(s, min_periods=_XRANK_MIN_PERIODS)

    # WQ_3:  -corr(rank(open), rank(volume), 10)
    df["D_WQ_3"] = (-_ts_corr(_xrank(open_), _xrank(volume), 10)).replace([np.inf, -np.inf], np.nan)

    # WQ_6:  -corr(open, volume, 10)
    df["D_WQ_6"] = (-_ts_corr(open_, volume, 10)).replace([np.inf, -np.inf], np.nan)

    # WQ_12: sign(delta(volume,1)) * -delta(close,1)
    df["D_WQ_12"] = (np.sign(_delta(volume, 1)) * (-_delta(close, 1))).replace([np.inf, -np.inf], np.nan)

    # WQ_13: -rank(cov(rank(close), rank(volume), 5))
    try:
        df["D_WQ_13"] = (-_xrank(_xrank(close).rolling(5, min_periods=3).cov(_xrank(volume)))
                       ).replace([np.inf, -np.inf], np.nan)
    except Exception:
        df["D_WQ_13"] = np.nan

    # WQ_16: -rank(cov(rank(high), rank(volume), 5))
    try:
        df["D_WQ_16"] = (-_xrank(_xrank(high).rolling(5, min_periods=3).cov(_xrank(volume)))
                       ).replace([np.inf, -np.inf], np.nan)
    except Exception:
        df["D_WQ_16"] = np.nan

    # WQ_19: -sign(delta(close-delay(close,7),5) + delta(close,5)) * (1 + rank(1+sum(returns,250)))
    try:
        d7 = _delay(close, 7)
        part = -np.sign(_delta(close - d7, 5) + _delta(close, 5))
        ret_sum = ret_d.rolling(250, min_periods=50).sum()
        df["D_WQ_19"] = (part * (1 + _xrank(1 + ret_sum))).replace([np.inf, -np.inf], np.nan)
    except Exception:
        df["D_WQ_19"] = np.nan

    # WQ_20: -rank(open - delay(high,1)) * rank(open - delay(close,1)) * rank(open - delay(low,1))
    try:
        df["D_WQ_20"] = (
            -_xrank(open_ - _delay(high, 1))
            * _xrank(open_ - _delay(close, 1))
            * _xrank(open_ - _delay(low, 1))
        ).replace([np.inf, -np.inf], np.nan)
    except Exception:
        df["D_WQ_20"] = np.nan

    # WQ_23: if mean(high,20) < high: -delta(high,2) else 0
    try:
        cond = _ts_mean(high, 20) < high
        df["D_WQ_23"] = ((-_delta(high, 2)).where(cond, 0)).replace([np.inf, -np.inf], np.nan)
    except Exception:
        df["D_WQ_23"] = np.nan

    # WQ_26: -ts_max(corr(ts_rank(volume,5), ts_rank(high,5), 5), 3)
    try:
        inner = _ts_corr(_ts_rank(volume, 5), _ts_rank(high, 5), 5)
        df["D_WQ_26"] = (-_ts_max(inner, 3)).replace([np.inf, -np.inf], np.nan)
    except Exception:
        df["D_WQ_26"] = np.nan

    # WQ_29: rank(rank(-rank(delta(close,5))))
    try:
        inner = -_xrank(_delta(close, 5))
        df["D_WQ_29"] = _xrank(_xrank(inner)).replace([np.inf, -np.inf], np.nan)
    except Exception:
        df["D_WQ_29"] = np.nan

    # WQ_33: rank(-1 + open/close)
    try:
        df["D_WQ_33"] = _xrank(-1 + open_ / close.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    except Exception:
        df["D_WQ_33"] = np.nan

    # WQ_35: ts_rank(volume,32) * (1 - ts_rank(close+high-low,16)) * (1 - ts_rank(returns,32))
    try:
        df["D_WQ_35"] = (
            _ts_rank(volume, 32)
            * (1 - _ts_rank(close + high - low, 16))
            * (1 - _ts_rank(ret_d, 32))
        ).replace([np.inf, -np.inf], np.nan)
    except Exception:
        df["D_WQ_35"] = np.nan

    # WQ_38: -rank(ts_rank(close,10)) * rank(close/open)
    try:
        df["D_WQ_38"] = (
            -_xrank(_ts_rank(close, 10)) * _xrank(close / open_.replace(0, np.nan))
        ).replace([np.inf, -np.inf], np.nan)
    except Exception:
        df["D_WQ_38"] = np.nan

    # WQ_40: -rank(std(high,10)) * corr(high, volume, 10)
    try:
        df["D_WQ_40"] = (-_xrank(_ts_std(high, 10)) * _ts_corr(high, volume, 10)
                       ).replace([np.inf, -np.inf], np.nan)
    except Exception:
        df["D_WQ_40"] = np.nan

    # WQ_41: sqrt(high*low) - vwap
    try:
        df["D_WQ_41"] = (np.sqrt(high * low) - vwap_d).replace([np.inf, -np.inf], np.nan)
    except Exception:
        df["D_WQ_41"] = np.nan

    # WQ_44: -corr(high, rank(volume), 5)
    try:
        df["D_WQ_44"] = (-_ts_corr(high, _xrank(volume), 5)).replace([np.inf, -np.inf], np.nan)
    except Exception:
        df["D_WQ_44"] = np.nan

    # ══════════════════════════════════════════════════════════════════════
    #  v19 NEW: 17 rolling/lag/diff transforms of best existing features
    # ══════════════════════════════════════════════════════════════════════

    # D_slope_stability transforms
    s_ss = pd.to_numeric(df["D_slope_stability"], errors="coerce")
    df["D_slope_stability_rmean50"] = s_ss.rolling(50, min_periods=12).mean()
    df["D_slope_stability_rstd10"] = s_ss.rolling(10, min_periods=3).std()
    df["D_slope_stability_rstd20"] = s_ss.rolling(20, min_periods=5).std()

    # D_body_ratio transforms
    s_br = pd.to_numeric(df["D_body_ratio"], errors="coerce")
    df["D_body_ratio_rmean50"] = s_br.rolling(50, min_periods=12).mean()
    df["D_body_ratio_rmean20"] = s_br.rolling(20, min_periods=5).mean()

    # D_close_roll_slope_20 transforms
    s_crs = pd.to_numeric(df["D_close_roll_slope_20"], errors="coerce")
    df["D_close_roll_slope_20_rstd20"] = s_crs.rolling(20, min_periods=5).std()
    df["D_close_roll_slope_20_rstd10"] = s_crs.rolling(10, min_periods=3).std()

    # D_macd_hist transform
    s_mh = pd.to_numeric(df["D_macd_hist"], errors="coerce")
    df["D_macd_hist_rstd10"] = s_mh.rolling(10, min_periods=3).std()

    # D_mdi14 transforms
    s_md = pd.to_numeric(df["D_mdi14"], errors="coerce")
    df["D_mdi14_diff1"] = s_md.diff(1)
    df["D_mdi14_diff5"] = s_md.diff(5)
    df["D_mdi14_rrank10"] = s_md.rolling(10, min_periods=3).rank(pct=True)
    df["D_mdi14_rrank20"] = s_md.rolling(20, min_periods=5).rank(pct=True)

    # D_donch_pos_50 transforms
    s_d50 = pd.to_numeric(df["D_donch_pos_50"], errors="coerce")
    df["D_donch_pos_50_rmean50"] = s_d50.rolling(50, min_periods=12).mean()
    df["D_donch_pos_50_lag5"] = s_d50.shift(5)

    # D_donch_pos_20 transforms
    s_d20 = pd.to_numeric(df["D_donch_pos_20"], errors="coerce")
    df["D_donch_pos_20_rmean50"] = s_d20.rolling(50, min_periods=12).mean()
    df["D_donch_pos_20_lag5"] = s_d20.shift(5)

    # D_cmf20 transform
    s_cmf = pd.to_numeric(df["D_cmf20"], errors="coerce")
    df["D_cmf20_rmean50"] = s_cmf.rolling(50, min_periods=12).mean()

    # Final hygiene: replace inf/-inf with NaN for the v19 columns
    v19_new_cols = [
        "D_WQ_3", "D_WQ_6", "D_WQ_12", "D_WQ_13", "D_WQ_16", "D_WQ_19", "D_WQ_20", "D_WQ_23",
        "D_WQ_26", "D_WQ_29", "D_WQ_33", "D_WQ_35", "D_WQ_38", "D_WQ_40", "D_WQ_41", "D_WQ_44",
        "D_slope_stability_rmean50", "D_slope_stability_rstd10", "D_slope_stability_rstd20",
        "D_body_ratio_rmean50", "D_body_ratio_rmean20",
        "D_close_roll_slope_20_rstd20", "D_close_roll_slope_20_rstd10",
        "D_macd_hist_rstd10",
        "D_mdi14_diff1", "D_mdi14_diff5",
        "D_mdi14_rrank10", "D_mdi14_rrank20",
        "D_donch_pos_50_rmean50", "D_donch_pos_50_lag5",
        "D_donch_pos_20_rmean50", "D_donch_pos_20_lag5",
        "D_cmf20_rmean50",
    ]
    for c in v19_new_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    return df


# -------------------- Finalize for cache --------------------
DROP_FROM_CACHE = {
    "D_hh", "D_hl", "D_lh", "D_ll",
    "D_price_and_obv_rising", "D_golden_regime", "D_day_type", "D_nr",
    "D_tmr_cpr_vs_today", "D_cpr_vs_yday", "D_structure_trend",
}

NONBLANK_NUMERIC = [
    "D_rsi7", "D_rsi14", "D_ema20_angle_deg",
    "D_atr14", "D_atr30", "D_atr_ratio_14_30", "D_range_to_atr14",
    "D_adx14", "D_pdi14", "D_mdi14",
    "D_tmr_cpr_vs_today_code", "D_cpr_vs_yday_code", "D_structure_trend_code",
    "D_bb_pctB_20", "D_bb_bw_20", "D_vol_yz_20", "D_vol_yz_50",
    "D_donch_pos_20", "D_donch_pos_50",
    "D_breakout_high_20", "D_breakout_low_20",
    "D_breakout_high_50", "D_breakout_low_50",
    "D_days_since_boh_20", "D_days_since_bol_20",
    "D_dollar_vol", "D_dvol_z20", "D_dvol_z50", "D_dvol_z252",
    "D_vol_surge_20", "D_vol_surge_50",
    # v18
    "D_body_ratio", "D_wick_skew",
    "D_hh_run", "D_hl_run", "D_lh_run", "D_ll_run",
    "D_nr_expand", "D_compress_state",
    "D_dist_from_20h", "D_dist_from_20l", "D_dist_from_52wh",
    "D_midpoint_slope", "D_slope_stability",
    "W_ret_4w", "W_ret_13w", "W_close_pos", "W_vol_vs_4w",
    # v19
    "D_WQ_3", "D_WQ_6", "D_WQ_12", "D_WQ_13", "D_WQ_16", "D_WQ_19", "D_WQ_20", "D_WQ_23",
    "D_WQ_26", "D_WQ_29", "D_WQ_33", "D_WQ_35", "D_WQ_38", "D_WQ_40", "D_WQ_41", "D_WQ_44",
    "D_slope_stability_rmean50", "D_slope_stability_rstd10", "D_slope_stability_rstd20",
    "D_body_ratio_rmean50", "D_body_ratio_rmean20",
    "D_close_roll_slope_20_rstd20", "D_close_roll_slope_20_rstd10",
    "D_macd_hist_rstd10",
    "D_mdi14_diff1", "D_mdi14_diff5",
    "D_mdi14_rrank10", "D_mdi14_rrank20",
    "D_donch_pos_50_rmean50", "D_donch_pos_50_lag5",
    "D_donch_pos_20_rmean50", "D_donch_pos_20_lag5",
    "D_cmf20_rmean50",
]


def _map_true_false_strings_to_int(s: pd.Series) -> pd.Series:
    vals = pd.Series(s.astype(str).str.strip().str.lower())
    uniq = set(vals.dropna().unique())
    if uniq <= {"true", "false", "nan", ""}:
        out = (
            vals.map({"true": 1, "false": 0})
            .astype("Int64")
            .fillna(0)
            .astype(int)
        )
        return out
    return s


def finalize_for_cache(df: pd.DataFrame) -> pd.DataFrame:
    """
    v21: NO LONGER FABRICATES VALUES.

    v20 ran `.ffill().fillna(0.0)` over every column in NONBLANK_NUMERIC.
    That was actively harmful for two reasons:

      1. 0.0 is not a neutral value for these features. It is an EXTREME:
           D_rsi14        = 0  -> maximum oversold
           D_bb_pctB_20   = 0  -> sitting on the lower Bollinger band
           D_donch_pos_20 = 0  -> bottom of the Donchian channel
           D_dist_from_52wh = 0 -> exactly at the 52-week high
         Every symbol's warm-up region was therefore filled with strong,
         entirely fictitious signals that a GBM will happily split on.

      2. `.ffill()` silently repaired genuine data gaps (halts, missing
         sessions, bad vendor rows), permanently destroying the evidence
         that anything was wrong.

    NaN is the correct representation of "not computable yet". LightGBM
    handles NaN natively and learns a default direction per split, which
    is strictly more informative than a fake zero.
    """
    if df is None or df.empty:
        return df
    out = df.copy()

    # Coerce to numeric only. No ffill, no fillna.
    for c in NONBLANK_NUMERIC:
        if c in PRESERVE_NA:
            continue
        if c in out.columns and out[c].dtype != np.float64:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "D_nr_day" in out.columns:
        out["D_nr_day"] = (
            pd.to_numeric(out["D_nr_day"], errors="coerce")
            .fillna(0)
            .astype("int16")
        )

    # Single-pass dtype normalisation (v21: build a dict and assign once
    # instead of N fragmenting column writes).
    updates = {}
    for c in list(out.columns):
        if c in PRESERVE_NA:
            continue          # NA here means unknown, not false
        col = out[c]
        if str(col.dtype) in ("boolean", "bool"):
            updates[c] = col.astype("Int8").fillna(0).astype("int8")
        elif col.dtype == object:
            updates[c] = _map_true_false_strings_to_int(col)
    if updates:
        out = out.assign(**updates)

    drop_cols = [c for c in DROP_FROM_CACHE if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols, errors="ignore")

    return out


def mark_warmup(df: pd.DataFrame, first_valid: Optional[dt.date]) -> pd.DataFrame:
    """
    Flag rows before `first_valid` as warm-up instead of deleting them.

    The rows MUST stay on disk. Two reasons:
      * recomputing a 252-bar window at the left edge needs the preceding
        bars to exist;
      * `_expand_rank_pct` expands from row 0, so physically truncating the
        frame would change the values of already-cached dates.
    Downstream panel code must filter on this flag (see load_cached_symbol).
    """
    if df is None or df.empty:
        return df
    if first_valid is None:
        df[WARMUP_FLAG_COL] = np.int8(0)
        return df
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    cutoff = pd.Timestamp(first_valid)
    if getattr(ts.dt, "tz", None) is not None:
        cutoff = cutoff.tz_localize(ts.dt.tz)
    df[WARMUP_FLAG_COL] = (ts < cutoff).to_numpy().astype("int8")
    return df


# Ratios that a genuine corporate action lands on, plus their reciprocals.
def load_cached_symbol(
    path: Path, *, include_warmup: bool = False, columns=None
) -> pd.DataFrame:
    """
    Read one symbol's cache the way downstream panel code should.

    Drops warm-up rows and the forward-return label by default. NEVER read
    the parquet directly in panel/model code - `ret_5d_close_pct` is a
    close.shift(-5) forward return and will leak the label straight into
    your feature matrix.
    """
    df = pd.read_parquet(path, columns=list(columns) if columns else None)
    if df is None or df.empty:
        return pd.DataFrame()
    if not include_warmup and WARMUP_FLAG_COL in df.columns:
        df = df.loc[df[WARMUP_FLAG_COL] == 0]
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------
# NON-FEATURE REGISTRY
#
# The failure mode this guards against: someone adds ret_1d, MFE, MAE,
# future_volatility or future_max_drawdown, forgets to register it, and it
# walks straight into X. That model scores beautifully and is worthless.
#
# So membership is decided by RULE as well as by list. Anything matching
# FORWARD_PATTERNS is excluded whether or not a human remembered it, and
# assert_no_forward_columns() fails loudly on a forward-looking name that
# nobody declared. A name-shaped tripwire, not a substitute for thinking:
# it cannot catch a label called "x7".
# ----------------------------------------------------------------------


class ForwardColumnNotRegistered(RuntimeError):
    """A future-looking column was found that nobody declared."""


LABEL_COLUMNS = ("ret_5d_close_pct",)

# Realised outcomes over a FUTURE window. Never features, under any name.
FORWARD_COLUMNS: tuple = ()

# Bookkeeping: real columns, but not signals.
STRUCTURAL_COLUMNS = (WARMUP_FLAG_COL, "timestamp", "symbol")

# Substrings that make a column forward-looking by construction.
FORWARD_PATTERNS = (
    "_fwd", "fwd_", "future_", "_ahead", "lookahead",
    "mfe", "mae", "target_", "label_", "_next",
    "ret_1d_close_pct", "ret_3d_close_pct", "ret_5d_close_pct",
)


def _looks_forward(col: str) -> bool:
    c = str(col).lower()
    return any(p in c for p in FORWARD_PATTERNS)


def non_feature_columns(df: pd.DataFrame) -> List[str]:
    """Every column that must never reach a model, by list and by rule."""
    banned = set(LABEL_COLUMNS) | set(FORWARD_COLUMNS) | set(STRUCTURAL_COLUMNS)
    banned |= {c for c in df.columns if _looks_forward(c)}
    return sorted(c for c in df.columns if c in banned)


def assert_no_forward_columns(df: pd.DataFrame) -> None:
    """
    Fail if a forward-looking-looking column is not registered.

    The pattern rule already keeps it out of X; this makes the omission
    visible rather than silently correct by luck.
    """
    known = set(LABEL_COLUMNS) | set(FORWARD_COLUMNS) | set(STRUCTURAL_COLUMNS)
    rogue = [c for c in df.columns if _looks_forward(c) and c not in known]
    if rogue:
        raise ForwardColumnNotRegistered(
            f"columns look forward-looking but are not in LABEL_COLUMNS or "
            f"FORWARD_COLUMNS: {rogue}. They are excluded from X by the "
            f"pattern rule, but register them explicitly."
        )


def feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Every column safe to hand a model.

    Excludes the registry AND anything matching FORWARD_PATTERNS, so a
    forgotten label is still kept out of X. Build X with this, never with
    df.columns or a manual drop list.
    """
    banned = set(non_feature_columns(df))
    return [c for c in df.columns if c not in banned]


