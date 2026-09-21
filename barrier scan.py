#!/usr/bin/env python3
"""
barrier_scan.py - what barriers are empirically FEASIBLE, from your own data.

    python barrier_scan.py --root <cache_root>
    python barrier_scan.py --root <cache_root> --atr
    python barrier_scan.py --root <cache_root> --horizon 10

WHAT CHANGED IN v2, AND WHY
===========================
v1 had two faults, one of them the kind I keep warning about.

  1. It called P(TP touched AND SL not touched) by the name `tp_pct`, which
     reads as P(TP first). Those are different quantities. MFE and MAE are
     both extremes over the window and neither records sequence, so v1 could
     not distinguish "+6% on day 1 then -4% on day 4" from "-4% on day 1 then
     +6% on day 4" - opposite bracket outcomes, identical MFE/MAE.

  2. Its expectancy line carried a comment saying unresolved episodes were
     "approximated by their realised mean return", and then did not add it.
     A comment asserting a property the code does not have is worse than no
     comment.

The fix for (1) is not a better approximation. It is to stop approximating:
this scanner reads the bars, so it can walk the path forward and find which
barrier was touched FIRST, exactly as the panel's path labels do. v1 threw
the path away and kept only the extremes. v2 keeps the running high and
running low per session, which is enough to resolve first-touch for every
bracket on the grid at once.

WHAT REMAINS GENUINELY UNKNOWABLE
---------------------------------
Daily bars do not record whether the high or the low came first WITHIN a
session. When both barriers are crossed on the same day the order cannot be
recovered from this data, at any level of effort. So every bracket is
reported as a RANGE:

    P(TP first) lower bound   same-day ties counted as SL
    P(TP first) upper bound   same-day ties counted as TP

If those two numbers are close, the bracket is well determined by daily bars.
If they are far apart, that bracket needs intraday data and no cleverness
here will substitute.

HOW TO USE IT
-------------
This finds the FEASIBLE target space. It does not choose for you, and the
highest-expectancy row is not "the answer" - that is how a scan becomes an
overfitting device. Pick using signal definition, execution horizon,
liquidity, costs, sample size and stability, then FREEZE the choice before
you evaluate the model. Choosing barriers after seeing model results is
another researcher degree of freedom.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Round-trip cost in basis points, NSE equity delivery.
#
# Every bracket looked profitable-ish gross and none of them are. Until the
# payoff is net, the table ranks brackets by a number that does not exist.
# Default is deliberately on the pessimistic side of typical: STT on both
# legs, exchange transaction charges, SEBI fee, GST, stamp duty, brokerage,
# plus a slippage allowance. Override with --cost-bps once you have measured
# your own fills - the GTT bracket and the size you trade both matter.
DEFAULT_COST_BPS = 35.0

TP_GRID = [0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15]
SL_GRID = [0.015, 0.02, 0.03, 0.04, 0.05]
ATR_TP_GRID = [0.75, 1.0, 1.5, 2.0, 3.0]
ATR_SL_GRID = [0.5, 0.75, 1.0, 1.5]


def _liquid_symbols(root: Path, syms, min_turnover: float, window: int = 60):
    """
    Keep symbols whose recent median turnover clears the floor.

    Without this the scan samples the whole cache, so the MFE/MAE and ATR
    distributions are dominated by names the panel excludes. Barriers chosen
    from that sample describe a universe you do not trade.
    """
    if not min_turnover:
        return syms
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_quality import _paths
    out = []
    for s in syms:
        pq, _ = _paths(root, s)
        try:
            d = pd.read_parquet(pq, columns=["close", "volume"]).tail(window)
        except Exception:
            continue
        tv = (pd.to_numeric(d["close"], errors="coerce")
              * pd.to_numeric(d["volume"], errors="coerce")).dropna()
        if len(tv) >= window // 2 and float(tv.median()) >= min_turnover:
            out.append(s)
    return out


def load_episodes(root: Path, horizon: int, sample: int,
                  min_turnover: float = 0.0, with_dates: bool = False):
    """
    Per-episode RUNNING extremes over the forward window, not just the final
    extremes. Keeping the running series is what makes first-touch
    recoverable: the first session whose running high crosses a level is the
    session that level was first reached.

    Returns (run_hi, run_lo, entry_ret, atr_pct) where run_hi/run_lo are
    (n_episodes x horizon) arrays of cumulative max/min return from entry.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_quality import _paths, _read_ok, discover_symbols

    syms = [s for s in discover_symbols(root)
            if (_read_ok(_paths(root, s)[1]) or {}).get("series_kind", "equity")
            == "equity"]
    before = len(syms)
    syms = _liquid_symbols(root, syms, min_turnover)
    if min_turnover:
        print(f"  liquidity floor {min_turnover:,.0f}/day: "
              f"{len(syms)} of {before} symbols qualify")
    rng = np.random.default_rng(0)
    rng.shuffle(syms)
    syms = syms[:sample]

    HI, LO, RET, ATR, DATE = [], [], [], [], []
    for s in syms:
        pq, _ = _paths(root, s)
        try:
            d = pd.read_parquet(pq, columns=["timestamp", "high", "low", "close"])
        except Exception:
            continue
        if len(d) < horizon + 80:
            continue
        c = pd.to_numeric(d["close"], errors="coerce").to_numpy(dtype="float64")
        h = pd.to_numeric(d["high"], errors="coerce").to_numpy(dtype="float64")
        l = pd.to_numeric(d["low"], errors="coerce").to_numpy(dtype="float64")
        n = len(c)

        tr = np.maximum(h[1:] - l[1:],
                        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        atr14 = np.concatenate([[np.nan],
                                pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()])

        idx = np.arange(60, n - horizon)
        if not len(idx):
            continue
        # (n_ep x horizon) forward windows
        off = np.arange(1, horizon + 1)
        fh = h[idx[:, None] + off]
        fl = l[idx[:, None] + off]
        base = c[idx][:, None]
        ok = np.isfinite(fh).all(axis=1) & np.isfinite(fl).all(axis=1) \
            & np.isfinite(base[:, 0]) & (base[:, 0] > 0)
        if not ok.any():
            continue
        HI.append(np.maximum.accumulate(fh[ok] / base[ok] - 1.0, axis=1))
        LO.append(np.minimum.accumulate(fl[ok] / base[ok] - 1.0, axis=1))
        RET.append(c[idx[ok] + horizon] / c[idx[ok]] - 1.0)
        ATR.append(atr14[idx[ok]] / c[idx[ok]])
        if with_dates:
            DATE.append(pd.to_datetime(d["timestamp"]).to_numpy()[idx[ok]])

    if not HI:
        raise RuntimeError("no usable episodes")
    dates = np.concatenate(DATE) if with_dates else None
    return (np.vstack(HI), np.vstack(LO), np.concatenate(RET),
            np.concatenate(ATR), dates)


def first_touch(run_hi, run_lo, tp_lvl, sl_lvl):
    """
    Session index of first crossing for each barrier, or horizon+1 if never.

    Both levels are arrays broadcast over episodes, so an ATR-scaled grid
    costs the same as a fixed-percentage one.
    """
    hz = run_hi.shape[1]
    hit_tp = run_hi >= tp_lvl[:, None]
    hit_sl = run_lo <= -sl_lvl[:, None]
    k_tp = np.where(hit_tp.any(axis=1), hit_tp.argmax(axis=1), hz)
    k_sl = np.where(hit_sl.any(axis=1), hit_sl.argmax(axis=1), hz)
    return k_tp, k_sl


def scan(run_hi, run_lo, fwd_ret, atr_pct, *, atr: bool = False,
         cost_bps: float = DEFAULT_COST_BPS) -> pd.DataFrame:
    hz = run_hi.shape[1]
    n_all = run_hi.shape[0]
    tps = ATR_TP_GRID if atr else TP_GRID
    sls = ATR_SL_GRID if atr else SL_GRID
    rows = []
    for tp in tps:
        for sl in sls:
            if atr:
                m = np.isfinite(atr_pct)
                tp_lvl, sl_lvl = tp * atr_pct[m], sl * atr_pct[m]
                rh, rl, fr = run_hi[m], run_lo[m], fwd_ret[m]
            else:
                tp_lvl = np.full(n_all, tp)
                sl_lvl = np.full(n_all, sl)
                rh, rl, fr = run_hi, run_lo, fwd_ret
            n = rh.shape[0]
            if not n:
                continue

            k_tp, k_sl = first_touch(rh, rl, tp_lvl, sl_lvl)
            touched_tp = k_tp < hz
            touched_sl = k_sl < hz
            tp_strictly_first = touched_tp & (k_tp < k_sl)
            sl_strictly_first = touched_sl & (k_sl < k_tp)
            same_day = touched_tp & touched_sl & (k_tp == k_sl)
            neither = ~touched_tp & ~touched_sl

            p_tp_lo = float(tp_strictly_first.sum()) / n
            p_tp_hi = float((tp_strictly_first | same_day).sum()) / n
            p_sl_lo = float(sl_strictly_first.sum()) / n
            p_none = float(neither.sum()) / n
            p_ambig = float(same_day.sum()) / n

            med_tp = float(np.median(tp_lvl))
            med_sl = float(np.median(sl_lvl))

            # Payoff INCLUDING the unresolved population, which exits at the
            # horizon at its realised return. v1 omitted this while claiming
            # otherwise.
            pay_none = float(fr[neither].mean()) if neither.any() else 0.0
            cost = cost_bps / 10000.0
            pay_lo = (p_tp_lo * med_tp
                      - (p_sl_lo + p_ambig) * med_sl
                      + p_none * pay_none) - cost
            pay_hi = (p_tp_hi * med_tp
                      - p_sl_lo * med_sl
                      + p_none * pay_none) - cost
            # How much the model must lift P(TP first) for this bracket to
            # break even NET. This is the number the whole engine has to beat.
            denom = med_tp + med_sl
            p_breakeven = ((cost + med_sl - p_none * pay_none) / denom
                           if denom > 0 else np.nan)

            k_tp_hit = k_tp[touched_tp]
            rows.append({
                "tp": f"{tp:.2f}" + ("a" if atr else ""),
                "sl": f"{sl:.3f}" + ("a" if atr else ""),
                "rr": round(med_tp / med_sl, 2),
                # tier 1: reachability - what daily bars establish outright
                "tp_touch": round(float(touched_tp.sum()) / n, 4),
                "sl_touch": round(float(touched_sl.sum()) / n, 4),
                "neither": round(p_none, 4),
                # tier 2: first-touch, as a RANGE over the unknowable ties
                "tp_first_lo": round(p_tp_lo, 4),
                "tp_first_hi": round(p_tp_hi, 4),
                "ambig": round(p_ambig, 4),
                # tier 3: timing
                "med_days_tp": float(np.median(k_tp_hit) + 1) if k_tp_hit.size else np.nan,
                "tp_1d": round(float((k_tp == 0).sum()) / n, 4),
                "tp_3d": round(float((k_tp <= 2).sum()) / n, 4),
                # payoff bounds, horizon-inclusive
                "pay_lo": round(pay_lo, 5),
                "pay_hi": round(pay_hi, 5),
                "p_breakeven": round(float(p_breakeven), 4),
                "lift_needed": round(float(p_breakeven) - p_tp_lo, 4),
                "n": n,
            })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--sample", type=int, default=250)
    ap.add_argument("--atr", action="store_true")
    ap.add_argument("--min-turnover", type=float, default=1e7,
                    help="liquidity floor; match your panel (0 disables)")
    ap.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS,
                    help="round-trip cost in bps")
    ap.add_argument("--stability", action="store_true",
                    help="re-scan per time period to test whether the ranking "
                         "is stable or fitted to noise")
    a = ap.parse_args()
    root = Path(a.root or os.environ.get("CACHE_DAILY_ROOT", "."))

    print(f"\n  sampling {a.sample} symbols, horizon {a.horizon} sessions...")
    hi, lo, fr, atrp, dates = load_episodes(
        root, a.horizon, a.sample, min_turnover=a.min_turnover,
        with_dates=a.stability)
    mfe, mae = hi[:, -1], lo[:, -1]
    print(f"  {len(mfe):,} episodes")
    print(f"  MFE  median {np.median(mfe):+.2%}  p75 {np.percentile(mfe,75):+.2%}"
          f"  p90 {np.percentile(mfe,90):+.2%}")
    print(f"  MAE  median {np.median(mae):+.2%}  p25 {np.percentile(mae,25):+.2%}"
          f"  p10 {np.percentile(mae,10):+.2%}")
    med_atr = float(np.nanmedian(atrp))
    print(f"  ATR% median {med_atr:.2%} -> a fixed 5% target is "
          f"{0.05/med_atr:.1f} median ATRs")

    res = scan(hi, lo, fr, atrp, atr=a.atr,
               cost_bps=a.cost_bps).sort_values("pay_lo", ascending=False)

    print(f"\n  TIER 1 - REACHABILITY (what daily bars establish outright)")
    print(f"  {'tp':>7}{'sl':>8}{'rr':>6}{'tp_touch':>10}{'sl_touch':>10}{'neither':>9}")
    for _, r in res.iterrows():
        print(f"  {r['tp']:>7}{r['sl']:>8}{r['rr']:>6.2f}{r['tp_touch']:>10.1%}"
              f"{r['sl_touch']:>10.1%}{r['neither']:>9.1%}")

    print(f"\n  TIER 2 - P(TP FIRST), as a RANGE over same-day ties")
    print(f"  {'tp':>7}{'sl':>8}{'lower':>8}{'upper':>8}{'ambig':>8}"
          f"{'pay_lo':>9}{'pay_hi':>9}")
    for _, r in res.iterrows():
        width = r["tp_first_hi"] - r["tp_first_lo"]
        mark = "  <- needs intraday" if width > 0.05 else ""
        print(f"  {r['tp']:>7}{r['sl']:>8}{r['tp_first_lo']:>8.1%}"
              f"{r['tp_first_hi']:>8.1%}{r['ambig']:>8.1%}"
              f"{r['pay_lo']:>9.4f}{r['pay_hi']:>9.4f}{mark}")

    print(f"\n  TIER 3 - TIMING")
    print(f"  {'tp':>7}{'sl':>8}{'med_days':>10}{'P(TP<=1d)':>11}{'P(TP<=3d)':>11}")
    for _, r in res.iterrows():
        md = f"{r['med_days_tp']:.1f}" if pd.notna(r["med_days_tp"]) else "n/a"
        print(f"  {r['tp']:>7}{r['sl']:>8}{md:>10}{r['tp_1d']:>11.1%}"
              f"{r['tp_3d']:>11.1%}")

    print(f"\n  TIER 4 - NET OF {a.cost_bps:.0f} bps ROUND-TRIP COST")
    print(f"  {'tp':>7}{'sl':>8}{'pay_lo':>10}{'pay_hi':>10}"
          f"{'P(TP) now':>11}{'breakeven':>11}{'lift needed':>13}")
    for _, r in res.iterrows():
        flag = "" if r["pay_hi"] > 0 else "   dead even at best"
        print(f"  {r['tp']:>7}{r['sl']:>8}{r['pay_lo']:>10.4f}"
              f"{r['pay_hi']:>10.4f}{r['tp_first_lo']:>11.1%}"
              f"{r['p_breakeven']:>11.1%}{r['lift_needed']:>+13.1%}{flag}")
    print("  'lift needed' is how far the model must push P(TP first) above")
    print("  the unconditional rate just to break even. That is the bar.")

    if a.stability and dates is not None:
        print(f"\n  TIER 5 - STABILITY ACROSS TIME")
        print("  If the ranking reshuffles between periods, the scan is fitting")
        print("  noise and ANY choice from it is arbitrary.")
        dd = pd.to_datetime(dates)
        yrs = dd.year.to_numpy()
        edges = np.array_split(np.unique(yrs), 3)
        rank_by_period = {}
        for gi, grp in enumerate(edges, 1):
            m = np.isin(yrs, grp)
            if m.sum() < 5000:
                continue
            sub = scan(hi[m], lo[m], fr[m], atrp[m], atr=a.atr,
                       cost_bps=a.cost_bps)
            sub = sub.sort_values("pay_lo", ascending=False).reset_index(drop=True)
            key = sub["tp"] + "/" + sub["sl"]
            rank_by_period[f"{grp[0]}-{grp[-1]}"] = {k: i for i, k in enumerate(key)}
        if len(rank_by_period) >= 2:
            keys = sorted(set().union(*[set(v) for v in rank_by_period.values()]))
            print(f"\n  {'bracket':>16}" + "".join(f"{p:>12}" for p in rank_by_period))
            for k in keys:
                cells = "".join(f"{rank_by_period[p].get(k, -1)+1:>12}"
                                for p in rank_by_period)
                print(f"  {k:>16}{cells}")
            import itertools
            cors = []
            for p1, p2 in itertools.combinations(rank_by_period, 2):
                common = set(rank_by_period[p1]) & set(rank_by_period[p2])
                if len(common) > 3:
                    x = [rank_by_period[p1][k] for k in common]
                    y = [rank_by_period[p2][k] for k in common]
                    cors.append(float(pd.Series(x).corr(pd.Series(y), method="spearman")))
            if cors:
                mc = float(np.mean(cors))
                verdict = ("stable - the ranking means something" if mc > 0.7
                           else "UNSTABLE - treat any pick as arbitrary" if mc < 0.4
                           else "partly stable - be careful")
                print(f"\n  mean rank correlation across periods: {mc:+.2f}  ({verdict})")

    print("\n  HOW TO READ THIS")
    print("  * lower/upper bracket P(TP first): same-day ties are unknowable")
    print("    from daily bars. A wide gap means that bracket needs intraday.")
    print("  * pay_lo/pay_hi include the unresolved population exiting at the")
    print("    horizon, and are GROSS of brokerage, slippage, impact and tax.")
    print("  * A tp_touch below ~10% leaves too few positives to train on.")
    print("  * A +3% target that usually takes 5 sessions is a different trade")
    print("    from one that usually takes 1 - see TIER 3 before choosing.")
    print("  * Run with and without --atr. Disagreement means the fixed-percent")
    print("    label is largely measuring which stocks are volatile.")
    print("  * Do NOT take the top row. Choose on signal, horizon, liquidity,")
    print("    costs and stability - then FREEZE it before evaluating a model.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
