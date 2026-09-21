#!/usr/bin/env python3
"""
soul.py - the "soul of the stock" engine. Standalone; NOT in the daily automation.

    python soul.py regimes    --panel <panel_dir>
    python soul.py personality --panel <panel_dir>
    python soul.py card --symbol RELIANCE --date 2026-09-19

WHAT THIS IS
============
Two components, in the order you specified:

  REGIME       what state is this stock in right now
  PERSONALITY  what does THIS stock do from that state, given how little
               data we have about it

Deliberately NOT here yet: the analogue engine, the meta-model, the ensemble.
Those come after there is a measured baseline and a populated ledger. Adding
them now would mean ten moving parts and no way to attribute a result to any
of them.

THE TWO WAYS THIS COULD SILENTLY BREAK
--------------------------------------
1. LEAKAGE THROUGH PERSONALITY.
   "How does RELIANCE behave after a volatility squeeze" is computed from
   outcomes. If the estimate at date t includes outcomes that resolved after
   t, the model is being told its own answer. Every personality number here
   is computed EXPANDING and LAGGED: at date t it uses only episodes whose
   5-day outcome window had already closed by t. That costs the first year
   of history and it is not optional.

2. FAKE PRECISION THROUGH SMALL SAMPLES.
   A stock has maybe 40 squeeze episodes in its whole history. A raw per-stock
   hit rate on 40 observations has a standard error around 8 percentage
   points - wider than any edge worth trading. So nothing is estimated per
   stock alone. Every per-stock number is SHRUNK toward the cross-sectional
   rate in proportion to how little evidence there is:

       shrunk = (n * stock_rate + k * global_rate) / (n + k)

   with n the EFFECTIVE sample size, not the raw count. Overlapping 5-day
   outcome windows mean 5 consecutive episodes are close to 1 independent
   observation, so the raw count overstates evidence by up to 5x. n_eff
   divides it out.

   The result: 40 episodes gets you mostly the global rate, 400 gets you
   mostly the stock's own. Personality where it is earned, and nowhere else.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

LABEL_HORIZON = 5          # must match panel_build.LABEL_HORIZON
SHRINK_K = 50.0            # prior weight, in EFFECTIVE observations
MIN_EFF_N = 5.0            # below this, report the global rate and say so

# ----------------------------------------------------------------------
# REGIME
#
# Discovered from thresholds on the stock's OWN distribution, not absolute
# numbers. An ATR% of 4 is ordinary for a smallcap and extreme for a large
# cap, so a fixed cut-off encodes market-cap, not regime. Every input below
# is a trailing percentile of that symbol's own history.
# ----------------------------------------------------------------------
REGIME_INPUTS = {
    "trend": "D_ema20_angle_deg",
    "vol": "D_realvol_20",
    "vol_ratio": "D_realvol_ratio_20_60",
    "location": "D_pos_in_52w_range",
    "volume": "D_dvol_z20",
}

REGIMES = [
    "strong_uptrend", "weak_uptrend", "sideways",
    "weak_downtrend", "strong_downtrend",
    "vol_compression", "vol_expansion",
]


def _expanding_pct(s: pd.Series, min_periods: int = 252) -> pd.Series:
    """Percentile of each value within that symbol's own PRIOR history."""
    return s.expanding(min_periods=min_periods).rank(pct=True)


def label_regimes(panel: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Assign one regime per (symbol, date), from trailing percentiles only.

    Volatility state takes precedence over trend state: a compression or an
    expansion changes what a trend means, so it is reported rather than
    buried inside "weak uptrend".
    """
    need = [c for c in REGIME_INPUTS.values() if c in panel.columns]
    missing = [c for c in REGIME_INPUTS.values() if c not in panel.columns]
    if missing and verbose:
        print(f"  WARNING: regime inputs absent from panel: {missing}")
    if not need:
        raise RuntimeError("no regime inputs present in the panel")

    p = panel.sort_values(["symbol", "timestamp"]).copy()
    g = p.groupby("symbol", sort=False)
    pct = {}
    for key, col in REGIME_INPUTS.items():
        if col in p.columns:
            pct[key] = g[col].transform(_expanding_pct)

    trend = pct.get("trend")
    volr = pct.get("vol_ratio")
    loc = pct.get("location")

    regime = pd.Series("sideways", index=p.index, dtype="object")
    if trend is not None:
        regime = regime.mask(trend >= 0.80, "strong_uptrend")
        regime = regime.mask((trend >= 0.60) & (trend < 0.80), "weak_uptrend")
        regime = regime.mask((trend <= 0.20), "strong_downtrend")
        regime = regime.mask((trend > 0.20) & (trend <= 0.40), "weak_downtrend")
    if volr is not None:
        regime = regime.mask(volr <= 0.15, "vol_compression")
        regime = regime.mask(volr >= 0.85, "vol_expansion")

    # Unknown until the symbol has enough of its own history to rank against.
    if trend is not None:
        regime = regime.mask(trend.isna(), pd.NA)

    p["regime"] = regime
    if loc is not None:
        p["regime_location_pct"] = loc.round(3)
    if verbose:
        vc = p["regime"].value_counts(dropna=True)
        print(f"  regimes assigned over {int(vc.sum()):,} rows "
              f"({int(p['regime'].isna().sum()):,} unknown - insufficient history)")
        for k, v in vc.items():
            print(f"    {k:<18} {v:>9,}  {v/vc.sum():>6.1%}")
    return p


# ----------------------------------------------------------------------
# PERSONALITY
# ----------------------------------------------------------------------
def effective_n_conservative(raw_n: float, horizon: int = LABEL_HORIZON) -> float:
    """
    A CONSERVATIVE HEURISTIC, not a statistically valid effective sample size.

    Consecutive daily observations share most of their outcome window: for a
    5-day label, days t and t+1 overlap in 4 of 5 days. Dividing by the
    horizon assumes the worst case - that episodes arrive in unbroken daily
    runs and each block of `horizon` rows is one observation.

    WHAT IT IS NOT: a measurement of the dependence structure. Real
    autocorrelation varies by stock and regime, episodes cluster unevenly,
    and there are gaps between them. The true ESS for a given statistic can
    be materially different in either direction. A proper version would
    decluster episodes by temporal separation, or compute an ESS from the
    autocorrelation of the statistic being estimated.

    It is named for what it is so that nobody downstream mistakes it for a
    real ESS. It errs toward under-claiming evidence, which for a shrinkage
    weight is the safe direction: you end up nearer the cross-sectional rate
    than you strictly need to be, rather than further from it.
    """
    return float(raw_n) / float(max(horizon, 1))


# Back-compat alias; prefer the explicit name.
effective_n = effective_n_conservative


def shrink_mean(stock_mean: float, global_mean: float, n_eff: float,
                k: float = None) -> float:
    """
    Shrinkage for a CONTINUOUS quantity (return, MFE, MAE, duration).

    Same form as the rate version. A proper empirical-Bayes treatment would
    set k from the ratio of within-stock to between-stock variance rather
    than fixing it, which would shrink noisy quantities harder than stable
    ones. Fixed k is a deliberate simplification and is flagged as such.
    """
    k = SHRINK_K if k is None else k
    if not np.isfinite(stock_mean):
        return global_mean
    return (n_eff * stock_mean + k * global_mean) / (n_eff + k)


def shrink(stock_rate: float, global_rate: float, n_eff: float,
           k: float = SHRINK_K) -> float:
    """Pull a per-stock estimate toward the cross-section by its own weakness."""
    if not np.isfinite(stock_rate):
        return global_rate
    return (n_eff * stock_rate + k * global_rate) / (n_eff + k)


def build_personality(
    panel: pd.DataFrame,
    *,
    label: str = "label_touch",
    by: str = "regime",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Per (symbol, regime) behaviour, estimated POINT-IN-TIME and shrunk.

    At each date the estimate uses only episodes whose outcome window had
    already closed - enforced by lagging the label by the horizon before any
    cumulative sum. So the value attached to date t could genuinely have been
    computed on the morning of t.

    Returns one row per (symbol, timestamp, regime) with:
        n_raw        overlapping episodes seen so far
        n_eff        independent-equivalent observations
        rate_stock   this stock's own rate (noisy, shown for inspection)
        rate_global  cross-sectional rate in this regime, also point-in-time
        rate_shrunk  what to actually use
        weight       how much of the estimate is the stock's own
    """
    if label not in panel.columns:
        raise RuntimeError(f"panel has no {label}")
    if by not in panel.columns:
        raise RuntimeError(f"panel has no {by}; run `regimes` first")

    p = panel[["timestamp", "symbol", by, label]].copy()
    p = p.sort_values(["symbol", "timestamp"])

    # THE LEAKAGE GUARD: an outcome for date d is knowable only at d+horizon.
    # Shifting the label forward by the horizon within each symbol means every
    # cumulative statistic below sees only closed windows.
    y = pd.to_numeric(p[label], errors="coerce")
    p["_y_known"] = y.groupby(p["symbol"], sort=False).shift(LABEL_HORIZON)

    frames = []
    for reg, grp in p.groupby(by, sort=False, dropna=True):
        grp = grp.sort_values(["symbol", "timestamp"])
        yk = grp["_y_known"]
        g = yk.groupby(grp["symbol"], sort=False)
        # expanding sums EXCLUDING the current row
        n_raw = g.apply(lambda s: s.notna().cumsum().shift(1)).reset_index(level=0, drop=True)
        hits = g.apply(lambda s: s.fillna(0).cumsum().shift(1)).reset_index(level=0, drop=True)
        grp = grp.assign(n_raw=n_raw.reindex(grp.index),
                         _hits=hits.reindex(grp.index))

        # global rate for this regime, also expanding and also lagged
        grp = grp.sort_values("timestamp")
        gn = grp["_y_known"].notna().cumsum().shift(1)
        gh = grp["_y_known"].fillna(0).cumsum().shift(1)
        grp["rate_global"] = (gh / gn.replace(0, np.nan))

        grp["rate_stock"] = grp["_hits"] / grp["n_raw"].replace(0, np.nan)
        grp["n_eff"] = grp["n_raw"].apply(
        lambda v: effective_n_conservative(v) if pd.notna(v) else np.nan)
        grp["rate_shrunk"] = [
            shrink(rs, rg, ne) if pd.notna(rg) and pd.notna(ne) else np.nan
            for rs, rg, ne in zip(grp["rate_stock"], grp["rate_global"], grp["n_eff"])
        ]
        grp["weight"] = grp["n_eff"] / (grp["n_eff"] + SHRINK_K)
        grp["evidence"] = np.where(
            grp["n_eff"].fillna(0) < MIN_EFF_N, "insufficient",
            np.where(grp["n_eff"] < 20, "weak",
                     np.where(grp["n_eff"] < 60, "moderate", "strong")))
        frames.append(grp)

    out = pd.concat(frames, ignore_index=True)
    out = out[["timestamp", "symbol", by, "n_raw", "n_eff", "rate_stock",
               "rate_global", "rate_shrunk", "weight", "evidence"]]
    out = out.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    if verbose:
        latest = out.groupby("symbol").tail(1)
        ev = latest["evidence"].value_counts()
        print(f"  personality rows: {len(out):,}")
        print(f"  latest-state evidence across {len(latest):,} symbols:")
        for k in ("strong", "moderate", "weak", "insufficient"):
            n = int(ev.get(k, 0))
            print(f"    {k:<14} {n:>6}  {n/max(len(latest),1):>6.1%}")
        mw = float(latest["weight"].dropna().mean()) if latest["weight"].notna().any() else 0
        print(f"  mean weight on the stock's own rate: {mw:.1%}")
        print(f"    (the rest is the cross-sectional rate - that is the "
              f"shrinkage doing its job)")
    return out


# ----------------------------------------------------------------------
# MULTIDIMENSIONAL PERSONALITY
#
# Direction alone is a thin description. What a trader needs from "how does
# this stock behave here" is the shape of the move: how far it runs, how far
# it goes against you first, whether the target is reached before the stop,
# and how long it takes.
#
# Every dimension is estimated the same way as the rate: expanding, lagged by
# the horizon so only closed outcomes are used, and shrunk toward the
# cross-sectional value by n_eff. A stock with 8 independent squeeze episodes
# gets essentially the global MFE, not its own.
# ----------------------------------------------------------------------

PERSONALITY_DIMENSIONS = {
    # column               how to read it
    "label_touch":        ("rate", "P(+5% within 5d)"),
    "label_tp_before_sl": ("rate", "P(TP before SL)"),
    "label_fwd_ret_5d":   ("mean", "Expected 5d return"),
    "label_mfe_5d":       ("mean", "Expected MFE"),
    "label_mae_5d":       ("mean", "Expected MAE"),
    "label_days_to_tp":   ("mean", "Days to target"),
}


def _expanding_lagged(vals: pd.Series, symbols: pd.Series) -> tuple:
    """Prior-only (count, sum) per symbol, excluding the current row."""
    g = vals.groupby(symbols, sort=False)
    cnt = g.apply(lambda s: s.notna().cumsum().shift(1)).reset_index(level=0, drop=True)
    tot = g.apply(lambda s: s.fillna(0).cumsum().shift(1)).reset_index(level=0, drop=True)
    return cnt, tot


def build_personality_multi(
    panel: pd.DataFrame,
    *,
    by: str = "regime",
    dimensions: dict = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    One row per (symbol, timestamp, regime) with every dimension estimated
    point-in-time and shrunk. Wide, not long, so a card is one lookup.
    """
    dims = dimensions or PERSONALITY_DIMENSIONS
    have = {c: v for c, v in dims.items() if c in panel.columns}
    missing = [c for c in dims if c not in panel.columns]
    if missing and verbose:
        print(f"  WARNING: dimensions absent from panel: {missing}")
        print(f"    (label_tp_before_sl needs the path-dependent labels - "
              f"rebuild the panel)")
    if not have:
        raise RuntimeError("no personality dimensions present in the panel")

    keep = ["timestamp", "symbol", by] + list(have)
    p = panel[keep].sort_values(["symbol", "timestamp"]).copy()

    # leakage guard, applied once for every dimension
    for c in have:
        v = pd.to_numeric(p[c], errors="coerce")
        p[f"_k_{c}"] = v.groupby(p["symbol"], sort=False).shift(LABEL_HORIZON)

    frames = []
    for reg, grp in p.groupby(by, sort=False, dropna=True):
        grp = grp.sort_values(["symbol", "timestamp"]).copy()
        base_cnt = None
        for c, (kind, _) in have.items():
            kv = grp[f"_k_{c}"]
            cnt, tot = _expanding_lagged(kv, grp["symbol"])
            cnt = cnt.reindex(grp.index); tot = tot.reindex(grp.index)
            stock = tot / cnt.replace(0, np.nan)

            gsorted = grp.sort_values("timestamp")
            gcnt = gsorted[f"_k_{c}"].notna().cumsum().shift(1)
            gtot = gsorted[f"_k_{c}"].fillna(0).cumsum().shift(1)
            glob = (gtot / gcnt.replace(0, np.nan)).reindex(grp.index)

            n_eff = cnt.apply(lambda v: effective_n_conservative(v)
                              if pd.notna(v) else np.nan)
            fn = shrink if kind == "rate" else shrink_mean
            grp[f"{c}__est"] = [
                fn(sv, gv, ne) if pd.notna(gv) and pd.notna(ne) else np.nan
                for sv, gv, ne in zip(stock, glob, n_eff)
            ]
            grp[f"{c}__stock"] = stock
            grp[f"{c}__global"] = glob
            if base_cnt is None:
                base_cnt, grp["n_raw"], grp["n_eff"] = cnt, cnt, n_eff

        grp["weight"] = grp["n_eff"] / (grp["n_eff"] + SHRINK_K)
        grp["evidence"] = np.where(
            grp["n_eff"].fillna(0) < MIN_EFF_N, "insufficient",
            np.where(grp["n_eff"] < 20, "weak",
                     np.where(grp["n_eff"] < 60, "moderate", "strong")))
        frames.append(grp)

    out = pd.concat(frames, ignore_index=True)
    drop = [c for c in out.columns if c.startswith("_k_")] + list(have)
    out = out.drop(columns=drop, errors="ignore")
    out = out.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    if verbose:
        print(f"  personality rows: {len(out):,} | dimensions: {len(have)}")
        latest = out.groupby("symbol").tail(1)
        ev = latest["evidence"].value_counts()
        for k in ("strong", "moderate", "weak", "insufficient"):
            n = int(ev.get(k, 0))
            print(f"    {k:<14} {n:>6}  {n/max(len(latest),1):>6.1%}")
    return out


def card_multi(personality: pd.DataFrame, symbol: str,
               date: Optional[str] = None) -> str:
    """Full behaviour card. Historical conditional estimates, NOT confidence."""
    pe = personality.loc[personality["symbol"] == symbol]
    if date:
        pe = pe.loc[pe["timestamp"] <= pd.Timestamp(date)]
    if pe.empty:
        return f"{symbol}: no estimate available"
    r = pe.iloc[-1]

    L = [f"{symbol} - {pd.Timestamp(r['timestamp']).date()}", "",
         f"  Regime                 {r.get('regime')}", ""]
    for c, (kind, lab) in PERSONALITY_DIMENSIONS.items():
        col = f"{c}__est"
        if col not in r or pd.isna(r[col]):
            continue
        v = r[col]
        txt = f"{v:.1%}" if kind == "rate" else (
            f"{v:+.2%}" if "ret" in c or "mfe" in c or "mae" in c else f"{v:.1f}")
        own = r.get(f"{c}__stock")
        glb = r.get(f"{c}__global")
        detail = ""
        if pd.notna(own) and pd.notna(glb):
            f2 = (lambda x: f"{x:.1%}") if kind == "rate" else (
                lambda x: f"{x:+.2%}" if "ret" in c or "fe" in c or "ae" in c
                else f"{x:.1f}")
            detail = f"   (stock {f2(own)} / all {f2(glb)})"
        L.append(f"  {lab:<22} {txt}{detail}")

    L += ["",
          f"  Episodes               "
          f"{int(r['n_raw']) if pd.notna(r['n_raw']) else 0} raw -> "
          f"{r['n_eff']:.0f} effective (conservative)"
          if pd.notna(r.get("n_eff")) else "  Episodes               0",
          f"  Evidence               {r.get('evidence')}",
          f"  Weight on this stock   "
          + (f"{r['weight']:.0%}" if pd.notna(r.get("weight")) else "0%"),
          "",
          "  These are historical conditional estimates, not predictions and",
          "  not calibrated confidence. They earn that status only when the",
          "  ledger shows them holding out-of-sample.",
          "  n_eff is a conservative heuristic (raw/horizon), not a measured ESS."]
    return "\n".join(L)


def card(personality: pd.DataFrame, panel: pd.DataFrame,
         symbol: str, date: Optional[str] = None) -> str:
    """The human-readable state card for one stock on one date."""
    pe = personality.loc[personality["symbol"] == symbol]
    pa = panel.loc[panel["symbol"] == symbol]
    if pe.empty or pa.empty:
        return f"{symbol}: not in the panel"
    if date:
        d = pd.Timestamp(date)
        pe = pe.loc[pe["timestamp"] <= d]
        pa = pa.loc[pa["timestamp"] <= d]
    if pe.empty:
        return f"{symbol}: no personality estimate at that date"
    r = pe.iloc[-1]
    a = pa.iloc[-1]

    def num(col, fmt="{:+.2f}"):
        v = a.get(col)
        return fmt.format(v) if pd.notna(v) else "n/a"

    lines = [
        f"{symbol} - {pd.Timestamp(r['timestamp']).date()}",
        "",
        f"  Regime                 {r['regime']}",
        f"  Position in 52w range  {num('D_pos_in_52w_range', '{:.2f}')}",
        f"  Drawdown from peak     {num('D_drawdown_252', '{:+.1%}')}",
        f"  Realised vol (20d)     {num('D_realvol_20', '{:.1f}')}%",
        "",
        f"  P(+5% within 5d)       {r['rate_shrunk']:.1%}"
            if pd.notna(r["rate_shrunk"]) else "  P(+5% within 5d)       n/a",
        f"    this stock alone     "
            + (f"{r['rate_stock']:.1%}" if pd.notna(r["rate_stock"]) else "n/a"),
        f"    all stocks, regime   "
            + (f"{r['rate_global']:.1%}" if pd.notna(r["rate_global"]) else "n/a"),
        "",
        f"  Episodes seen          {int(r['n_raw']) if pd.notna(r['n_raw']) else 0}"
        f" raw -> {r['n_eff']:.0f} independent"
            if pd.notna(r["n_eff"]) else "  Episodes seen          0",
        f"  Evidence               {r['evidence']}",
        f"  Weight on this stock   "
            + (f"{r['weight']:.0%}" if pd.notna(r["weight"]) else "0%"),
        "",
        "  NOTE: this is a historical conditional rate, not a prediction, and",
        "  it is not calibrated. It becomes a confidence number only after the",
        "  ledger shows these rates holding out-of-sample.",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["regimes", "personality", "card"])
    ap.add_argument("--multi", action="store_true",
                    help="multidimensional personality (direction, magnitude, "
                         "MFE, MAE, TP-before-SL, duration)")
    ap.add_argument("--panel", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--date", default=None)
    a = ap.parse_args()

    root = Path(a.root or os.environ.get("CACHE_DAILY_ROOT", "."))
    pdir = Path(a.panel) if a.panel else root / "panel"
    ppq = pdir / "panel.parquet" if pdir.is_dir() else pdir
    soul_dir = ppq.parent / "soul"
    soul_dir.mkdir(parents=True, exist_ok=True)

    if a.cmd == "regimes":
        cols = ["timestamp", "symbol"] + [c for c in REGIME_INPUTS.values()]
        panel = pd.read_parquet(ppq)
        keep = [c for c in cols if c in panel.columns]
        out = label_regimes(panel[keep])
        out.to_parquet(soul_dir / "regimes.parquet", index=False)
        print(f"  -> {soul_dir / 'regimes.parquet'}")

    elif a.cmd == "personality":
        reg_p = soul_dir / "regimes.parquet"
        if not reg_p.exists():
            print("run `regimes` first", file=sys.stderr)
            return 2
        reg = pd.read_parquet(reg_p, columns=["timestamp", "symbol", "regime"])
        if a.multi:
            av = pd.read_parquet(ppq).columns
            cols = ["timestamp", "symbol"] + [c for c in PERSONALITY_DIMENSIONS
                                              if c in av]
            panel = pd.read_parquet(ppq, columns=cols)
            merged = panel.merge(reg, on=["timestamp", "symbol"], how="left")
            out = build_personality_multi(merged)
        else:
            panel = pd.read_parquet(ppq, columns=["timestamp", "symbol", "label_touch"])
            merged = panel.merge(reg, on=["timestamp", "symbol"], how="left")
            out = build_personality(merged)
        out.to_parquet(soul_dir / "personality.parquet", index=False)
        print(f"  -> {soul_dir / 'personality.parquet'}")

    else:
        if not a.symbol:
            print("card needs --symbol", file=sys.stderr)
            return 2
        pe = pd.read_parquet(soul_dir / "personality.parquet")
        print()
        if any(c.endswith("__est") for c in pe.columns):
            print(card_multi(pe, a.symbol, a.date))
        else:
            print(card(pe, pd.read_parquet(ppq), a.symbol, a.date))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
