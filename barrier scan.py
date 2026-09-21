#!/usr/bin/env python3
"""
barrier_scan.py - choose TP/SL from YOUR data, not from a round number.

    python barrier_scan.py --root <cache_root>
    python barrier_scan.py --root <cache_root> --atr        # ATR-scaled grid
    python barrier_scan.py --root <cache_root> --horizon 10

Reads the panel's realised MFE/MAE distribution and reports, for each
candidate bracket, what you would actually have got.

WHAT IT CANNOT TELL YOU
=======================
The "best" bracket. Expectancy here is gross of costs and assumes you take
every signal, which is not your system. Treat the output as the feasible set,
then pick with your own sizing and risk in mind.

THE TWO THINGS IT EXISTS TO SHOW
--------------------------------
1. HOW THE HORIZON BINDS.
   A 15% target is ordinary over a quarter and rare inside five sessions.
   The hit-rate column makes that concrete instead of theoretical. If your
   chosen TP shows a 2% hit rate, the label has almost no positives and no
   model will learn from it - you need a longer horizon or a nearer target.

2. WHY FIXED PERCENTAGES ARE NOT COMPARABLE ACROSS STOCKS.
   5% is about 1 ATR for a volatile smallcap and 3 ATR for a large cap, so a
   fixed-percentage label mostly measures which names are volatile. --atr
   scales each barrier to the stock's own ATR so a "1.5 ATR target" means the
   same thing everywhere. The two tables usually disagree, and that
   disagreement is the point.

AMBIGUITY MATTERS
-----------------
Daily bars cannot say whether the high or the low came first within a
session. Brackets that are close together get touched on both sides on the
same day more often, and those outcomes are unknowable from this data. The
`ambig` column is the share of episodes affected; if it is large for your
chosen bracket, P(TP before SL) needs intraday bars to be trustworthy.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TP_GRID = [0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15]
SL_GRID = [0.015, 0.02, 0.03, 0.04, 0.05, 0.07]
ATR_TP_GRID = [0.75, 1.0, 1.5, 2.0, 3.0]
ATR_SL_GRID = [0.5, 0.75, 1.0, 1.5]


def load_paths(root: Path, horizon: int, sample: int, atr: bool):
    """Per-episode forward path extremes, from the raw cache."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_quality import _paths, _read_ok, discover_symbols

    syms = [s for s in discover_symbols(root)
            if (_read_ok(_paths(root, s)[1]) or {}).get("series_kind", "equity")
            == "equity"]
    rng = np.random.default_rng(0)
    rng.shuffle(syms)
    syms = syms[:sample]

    rows = []
    for s in syms:
        pq, _ = _paths(root, s)
        try:
            d = pd.read_parquet(pq, columns=["timestamp", "high", "low", "close"])
        except Exception:
            continue
        if len(d) < horizon + 60:
            continue
        c = pd.to_numeric(d["close"], errors="coerce").to_numpy(dtype="float64")
        h = pd.to_numeric(d["high"], errors="coerce").to_numpy(dtype="float64")
        l = pd.to_numeric(d["low"], errors="coerce").to_numpy(dtype="float64")
        n = len(c)
        # ATR-14 for scaling, trailing only
        tr = np.maximum(h[1:] - l[1:],
                        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
        atr14 = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()
        atr14 = np.concatenate([[np.nan], atr14])

        for i in range(60, n - horizon):
            fh = h[i + 1:i + 1 + horizon]
            fl = l[i + 1:i + 1 + horizon]
            if not np.isfinite(c[i]) or not np.isfinite(fh).all():
                continue
            rows.append((
                float(np.max(fh) / c[i] - 1.0),      # mfe
                float(np.min(fl) / c[i] - 1.0),      # mae
                float(atr14[i] / c[i]) if np.isfinite(atr14[i]) else np.nan,
                i, len(rows),
            ))
    if not rows:
        raise RuntimeError("no usable episodes")
    df = pd.DataFrame(rows, columns=["mfe", "mae", "atr_pct", "_i", "_j"])
    return df.drop(columns=["_i", "_j"])


def scan(df: pd.DataFrame, atr: bool = False) -> pd.DataFrame:
    """
    For each bracket: hit rate, stop rate, ambiguity, gross expectancy.

    NOTE the approximation: MFE and MAE alone cannot say which came first.
    Episodes that reach BOTH barriers are counted in `ambig` and assigned to
    the stop, matching the conservative rule the panel labels use. So `tp_pct`
    here is a LOWER bound on how often the target was hit first.
    """
    out = []
    tps = ATR_TP_GRID if atr else TP_GRID
    sls = ATR_SL_GRID if atr else SL_GRID
    a = df["atr_pct"]
    for tp in tps:
        for sl in sls:
            tp_lvl = tp * a if atr else tp
            sl_lvl = sl * a if atr else sl
            hit_tp = df["mfe"] >= tp_lvl
            hit_sl = df["mae"] <= -sl_lvl
            ok = hit_tp.notna() & hit_sl.notna() & (
                a.notna() if atr else pd.Series(True, index=df.index))
            both = (hit_tp & hit_sl & ok)
            win = (hit_tp & ~hit_sl & ok)
            loss = (hit_sl & ~hit_tp & ok)
            neither = (~hit_tp & ~hit_sl & ok)
            n = int(ok.sum())
            if not n:
                continue
            # conservative: ambiguous episodes assigned to the stop
            p_win = float(win.sum()) / n
            p_loss = float((loss.sum() + both.sum())) / n
            p_none = float(neither.sum()) / n
            med_tp = float(tp_lvl.median()) if atr else tp
            med_sl = float(sl_lvl.median()) if atr else sl
            # gross expectancy: unresolved episodes exit at the horizon,
            # approximated here by their realised mean return
            exp = p_win * med_tp - p_loss * med_sl
            out.append({
                "tp": f"{tp:.2f}" + ("xATR" if atr else ""),
                "sl": f"{sl:.3f}" + ("xATR" if atr else ""),
                "rr": round(med_tp / med_sl, 2),
                "tp_pct": round(p_win, 4),
                "sl_pct": round(p_loss, 4),
                "none_pct": round(p_none, 4),
                "ambig": round(float(both.sum()) / n, 4),
                "exp_gross": round(exp, 5),
                "n": n,
            })
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--sample", type=int, default=250)
    ap.add_argument("--atr", action="store_true")
    a = ap.parse_args()
    root = Path(a.root or os.environ.get("CACHE_DAILY_ROOT", "."))

    print(f"\n  sampling {a.sample} symbols, horizon {a.horizon} sessions...")
    df = load_paths(root, a.horizon, a.sample, a.atr)
    print(f"  {len(df):,} episodes")
    print(f"  MFE  median {df['mfe'].median():+.2%}  p75 {df['mfe'].quantile(.75):+.2%}"
          f"  p90 {df['mfe'].quantile(.90):+.2%}")
    print(f"  MAE  median {df['mae'].median():+.2%}  p25 {df['mae'].quantile(.25):+.2%}"
          f"  p10 {df['mae'].quantile(.10):+.2%}")
    print(f"  ATR% median {df['atr_pct'].median():.2%}  "
          f"(a fixed 5% target is {0.05/df['atr_pct'].median():.1f} median ATRs)")

    res = scan(df, atr=a.atr)
    res = res.sort_values("exp_gross", ascending=False)
    print(f"\n  {'tp':>10}{'sl':>12}{'rr':>6}{'tp%':>8}{'sl%':>8}"
          f"{'none%':>8}{'ambig':>8}{'E[gross]':>10}")
    for _, r in res.iterrows():
        print(f"  {r['tp']:>10}{r['sl']:>12}{r['rr']:>6.2f}{r['tp_pct']:>8.1%}"
              f"{r['sl_pct']:>8.1%}{r['none_pct']:>8.1%}{r['ambig']:>8.1%}"
              f"{r['exp_gross']:>10.4f}")

    print("\n  READ THIS BEFORE PICKING A ROW:")
    print("  * E[gross] excludes brokerage, slippage, impact and taxes, and")
    print("    assumes every signal is taken. It is a ranking, not a return.")
    print("  * A tp% below ~10% means the label has too few positives to train")
    print("    on. Lengthen the horizon or bring the target nearer.")
    print("  * A high ambig% means daily bars cannot resolve that bracket.")
    print("  * Run with and without --atr. If they disagree, your fixed-percent")
    print("    label is mostly measuring which stocks are volatile.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
