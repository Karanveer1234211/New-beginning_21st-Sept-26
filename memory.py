#!/usr/bin/env python3
"""
memory.py - episodic memory and analogue search. Standalone; NOT in the daily job.

    python memory.py build --root <cache_root>
    python memory.py build --root <cache_root> --incremental
    python memory.py ask --symbol RELIANCE --root <cache_root>
    python memory.py ask --symbol TCS --date 2026-06-30 --scope cross

WHAT IT DOES
============
Stores a normalised state vector for every (symbol, session) and, when asked,
finds the historical episodes most similar to today - then reports what
actually happened next.

WHY THIS IS THE MOST DANGEROUS COMPONENT
========================================
"147 similar situations, 63% of them went up" is the most persuasive output
this whole system can produce, and it is wrong far more often than it looks.
Three failure modes, all handled explicitly below:

1. TEMPORAL CLUSTERING.
   Days 412, 413 and 414 are nearly the same state with nearly the same
   outcome window. Take the 147 nearest neighbours and you may be looking at
   eight independent episodes wearing a coat. DECLUSTERING enforces a minimum
   session gap between selected analogues within a symbol, so the count you
   see is closer to a count of distinct events.

2. LEAKAGE THROUGH THE FUTURE.
   An analogue is only usable if its outcome had already resolved when the
   query date arrived. Every search is bounded at `query_date - horizon`, so
   a neighbour whose 5-day window is still open cannot contribute.

3. SELF-MATCH.
   The nearest neighbour to today is yesterday, whose outcome overlaps today's
   almost entirely. An exclusion window around the query removes it.

WHAT THE CONFIDENCE SCORE IS, AND IS NOT
----------------------------------------
It combines effective sample size, how tight the neighbours are in state
space, and how much they agree on the outcome. It is a measure of EVIDENCE
STRENGTH, not a calibrated probability. It has never been checked against
realised outcomes, because that requires the prediction ledger to accumulate
first. Until it does, treat a high score as "this situation is well
precedented", not as "this will happen".
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

LABEL_HORIZON = 5

# The state vector. Deliberately small: a dozen axes that describe WHERE a
# stock is, not everything known about it. High-dimensional nearest-neighbour
# search degenerates - every point becomes equidistant - so more features
# would make matches worse, not better.
STATE_FEATURES = [
    "D_ema20_angle_deg",      # trend direction
    "D_realvol_20",           # volatility level
    "D_realvol_ratio_20_60",  # volatility regime change
    "D_rsi14",                # momentum position
    "D_adx14",                # trend strength
    "D_pos_in_52w_range",     # location in the yearly range
    "D_drawdown_252",         # distance below the peak
    "D_dist_from_52wh",       # distance from the high, in ATRs
    "D_bb_pctB_20",           # position in the volatility envelope
    "D_donch_pos_20",         # position in the recent channel
    "D_dvol_z20",             # volume vs its own recent norm
    "D_gap_pct",              # overnight move
]

OUTCOME_COLS = [
    "label_touch", "label_tp_before_sl", "label_fwd_ret_5d",
    "label_mfe_5d", "label_mae_5d", "label_days_to_tp",
]

# PIT NORMALISATION EPOCHS.
#
# v2 ranked every feature over the WHOLE panel, so a 2018 state's numerical
# representation depended on 2026 data. That is not a harmless monotone
# transform: the rank assigned to a value moves as history accumulates, and
# once representations move, the Euclidean ordering of neighbours moves with
# them. Which episodes are "nearest", which clear the distance cut, and
# therefore the reported outcome distribution - all future-contaminated.
#
# v3 freezes a normalisation per epoch from data STRICTLY BEFORE that epoch
# begins, and a query uses the epoch active at ITS OWN date. Both the query
# and every candidate are mapped through the same frozen transform, because
# distances between vectors in different spaces are meaningless.
#
# The cost: the state matrix cannot be precomputed once. Raw values are
# stored and normalised at query time, which is why the pool is narrowed
# before normalising.
EPOCH_FREQ = "YS"          # refresh the normalisation each calendar year
QUANTILE_GRID = 1000       # resolution of the frozen quantile edges

DEFAULT_K = 200            # candidate neighbours BEFORE declustering
MIN_SEPARATION = LABEL_HORIZON   # sessions between analogues of one symbol
# How close counts as "an analogue", expressed as a percentile of the pool's
# OWN distance distribution rather than a fraction of the theoretical maximum.
#
# The absolute version was wrong: in a 12-dimensional rank space the distance
# between two random points concentrates near sqrt(d/6), so nearly everything
# sits close to the maximum and a fixed fraction of it admits nothing. That is
# the curse of dimensionality doing what it always does. A percentile is
# scale-free and adapts to however the space actually fills up.
DISTANCE_PCTILE = 1.0      # keep neighbours inside the closest 1% of the pool


def memory_dir(panel_path: Path) -> Path:
    d = panel_path.parent / "memory"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ----------------------------------------------------------------------
# BUILD
# ----------------------------------------------------------------------
def build_memory(panel_path, *, verbose: bool = True) -> Path:
    """
    Store RAW state values plus a frozen, point-in-time normalisation per epoch.

    Two artefacts:
      state_raw.npy   raw feature values, never transformed
      edges.npz       per-epoch quantile edges, each computed only from data
                      STRICTLY BEFORE that epoch's start date

    Nothing here is normalised ahead of time, because the correct
    normalisation depends on WHEN the question is asked, not when the episode
    happened.
    """
    panel_path = Path(panel_path)
    md = memory_dir(panel_path)

    cols = ["timestamp", "symbol"] + STATE_FEATURES + OUTCOME_COLS
    try:
        p = pd.read_parquet(panel_path, columns=cols)
    except Exception:
        full = pd.read_parquet(panel_path)
        keep = [c for c in cols if c in full.columns]
        missing = [c for c in cols if c not in full.columns]
        if verbose and missing:
            print(f"  WARNING: absent from panel, skipping: {missing}")
        p = full[keep]

    feats = [c for c in STATE_FEATURES if c in p.columns]
    outs = [c for c in OUTCOME_COLS if c in p.columns]
    p = p.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    p["timestamp"] = pd.to_datetime(p["timestamp"]).dt.tz_localize(None)

    X = np.column_stack([
        pd.to_numeric(p[c], errors="coerce").to_numpy(dtype="float32")
        for c in feats
    ])
    usable = ~np.isnan(X).any(axis=1)
    X, p = X[usable], p.loc[usable].reset_index(drop=True)
    if verbose:
        print(f"  {len(p):,} episodes | {len(feats)} state axes | "
              f"{len(outs)} outcome columns")

    # One frozen normalisation per epoch, from prior data only.
    ts = p["timestamp"]
    starts = pd.date_range(ts.min().normalize(), ts.max().normalize(),
                           freq=EPOCH_FREQ)
    starts = pd.DatetimeIndex([ts.min()]).append(starts).unique().sort_values()
    qs = np.linspace(0, 100, QUANTILE_GRID + 1)
    edges, epoch_keys, skipped = [], [], 0
    for st in starts:
        prior = ts < st
        n_prior = int(prior.sum())
        if n_prior < 5000:
            skipped += 1
            continue
        e = np.column_stack([np.percentile(X[prior.to_numpy(), j], qs)
                             for j in range(X.shape[1])]).astype("float32")
        edges.append(e)
        epoch_keys.append(np.datetime64(st, "ns"))
    if not edges:
        raise RuntimeError("not enough history to freeze any normalisation")

    np.save(md / "state_raw.npy", X)
    np.savez(md / "edges.npz", edges=np.stack(edges),
             epochs=np.array(epoch_keys, dtype="datetime64[ns]"))
    p[["timestamp", "symbol"] + outs].to_parquet(md / "index.parquet", index=False)
    (md / "memory_meta.json").write_text(json.dumps({
        "built_at": dt.datetime.now().isoformat(),
        "episodes": int(len(p)),
        "state_features": feats,
        "outcome_columns": outs,
        "horizon_sessions": LABEL_HORIZON,
        "epochs": [str(pd.Timestamp(e).date()) for e in epoch_keys],
        "epochs_skipped_for_thin_history": skipped,
        "normalisation": "point-in-time quantile edges, frozen per epoch",
    }, indent=2), encoding="utf-8")
    if verbose:
        print(f"  {len(edges)} PIT normalisation epochs "
              f"({skipped} skipped - insufficient prior history)")
        print(f"  -> {md}")
    return md


def _normalise(vals: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map raw values onto [0,1] using frozen quantile edges."""
    out = np.empty_like(vals, dtype="float32")
    n = edges.shape[0] - 1
    for j in range(vals.shape[1]):
        out[:, j] = np.searchsorted(edges[:, j], vals[:, j], side="left") / n
    return np.clip(out, 0.0, 1.0)


def _epoch_for(q_time, epochs: np.ndarray) -> int:
    """Index of the latest epoch whose normalisation was frozen before q_time."""
    ok = np.where(epochs <= q_time)[0]
    if not len(ok):
        raise RuntimeError(
            "query predates every frozen normalisation - no point-in-time "
            "transform was available that early")
    return int(ok[-1])


# ----------------------------------------------------------------------
# SEARCH
# ----------------------------------------------------------------------
def _decluster(order: np.ndarray, ts: np.ndarray, syms: np.ndarray,
               min_sep_days: int) -> np.ndarray:
    """
    Keep the closest analogue in each cluster of near-identical episodes.

    Walking the neighbours from nearest outward, an episode is accepted only
    if it is at least `min_sep_days` from every already-accepted episode of
    the SAME symbol. Different symbols are independent by construction.

    This is what stops 147 matches that are really eight events from being
    counted as 147.
    """
    kept: List[int] = []
    last: Dict[str, List[np.datetime64]] = {}
    for i in order:
        s = syms[i]
        t = ts[i]
        prior = last.get(s, [])
        if any(abs((t - u) / np.timedelta64(1, "D")) < min_sep_days for u in prior):
            continue
        kept.append(i)
        last.setdefault(s, []).append(t)
    return np.array(kept, dtype=int)


def ask(
    panel_path,
    symbol: str,
    *,
    date: Optional[str] = None,
    scope: str = "self",          # self | cross | both
    k: int = DEFAULT_K,
    min_separation: int = MIN_SEPARATION,
    verbose: bool = True,
) -> dict:
    """
    Find the historical episodes most like this one, and report what followed.

    scope="self"  only this stock's own history - "what does THIS stock do"
    scope="cross" every stock - more evidence, less specificity
    scope="both"  reports the two separately, because they answer different
                  questions and averaging them hides the disagreement
    """
    panel_path = Path(panel_path)
    md = memory_dir(panel_path)
    if not (md / "state_raw.npy").exists():
        raise RuntimeError("memory not built - run `build` first")

    Xraw = np.load(md / "state_raw.npy")
    ez = np.load(md / "edges.npz", allow_pickle=False)
    all_edges, epochs = ez["edges"], ez["epochs"]
    idx = pd.read_parquet(md / "index.parquet")
    meta = json.loads((md / "memory_meta.json").read_text(encoding="utf-8"))
    horizon = int(meta.get("horizon_sessions", LABEL_HORIZON))
    outs = [c for c in meta["outcome_columns"] if c in idx.columns]

    ts = idx["timestamp"].to_numpy(dtype="datetime64[ns]")
    syms = idx["symbol"].to_numpy()

    mask_sym = syms == symbol
    if not mask_sym.any():
        raise RuntimeError(f"{symbol} not in memory")
    if date:
        q_t = np.datetime64(pd.Timestamp(date), "ns")
        cand = np.where(mask_sym & (ts <= q_t))[0]
    else:
        cand = np.where(mask_sym)[0]
    if not len(cand):
        raise RuntimeError(f"{symbol} has no episode at or before {date}")
    q = cand[np.argmax(ts[cand])]
    q_time = ts[q]

    # THE LEAKAGE BOUND, IN SESSIONS NOT CALENDAR DAYS.
    #
    # v2 subtracted horizon+1 CALENDAR days. The labels span 5 TRADING
    # sessions, which is not the same thing: a Thursday query minus six
    # calendar days lands on Friday, but five sessions earlier is the
    # previous Thursday. Weekends and holidays widen the gap, and every day
    # of slack admits an analogue whose outcome had not yet resolved.
    #
    # The session calendar comes from the memory index itself - the sorted
    # unique dates on which anything in the panel traded.
    sessions = np.unique(ts)
    qpos = int(np.searchsorted(sessions, q_time, side="right")) - 1
    cut_pos = qpos - horizon
    if cut_pos < 0:
        return {"symbol": symbol, "as_of": str(pd.Timestamp(q_time).date()),
                "error": "query sits inside the first few sessions of history"}
    cutoff = sessions[cut_pos]
    usable = ts <= cutoff
    if scope == "self":
        usable &= (syms == symbol)
    elif scope == "cross":
        usable &= (syms != symbol)
    pool = np.where(usable)[0]
    if len(pool) < 20:
        return {"symbol": symbol, "as_of": str(pd.Timestamp(q_time).date()),
                "error": f"only {len(pool)} usable prior episodes"}

    # Normalise the query AND the pool through the SAME frozen transform -
    # the one available at the query date. Vectors normalised under different
    # transforms are not comparable, which is why this cannot be precomputed.
    ei = _epoch_for(q_time, epochs)
    edges = all_edges[ei]
    Xp = _normalise(Xraw[pool], edges)
    xq = _normalise(Xraw[q:q + 1], edges)[0]
    epoch_used = str(pd.Timestamp(epochs[ei]).date())

    d = np.sqrt(((Xp - xq) ** 2).sum(axis=1))
    # The cut is set from THIS query's own distance distribution, so it means
    # the same thing whatever the dimensionality or how densely the space is
    # populated around this particular state.
    cut = float(np.percentile(d, DISTANCE_PCTILE))
    take = min(k * 5, len(pool))
    near = pool[np.argsort(d)[:take]]
    near_d = np.sort(d)[:take]

    kept = _decluster(np.arange(len(near)), ts[near], syms[near], min_separation)
    kept = kept[:k]
    sel = near[kept]
    sel_d = near_d[kept]

    close = sel_d <= cut
    n_eff = int(close.sum())

    rows = idx.iloc[sel[close]] if n_eff else idx.iloc[[]]
    summary = {}
    for c in outs:
        v = pd.to_numeric(rows[c], errors="coerce").dropna()
        if v.empty:
            continue
        summary[c] = {
            "n": int(len(v)),
            "mean": float(v.mean()),
            "p25": float(v.quantile(0.25)),
            "median": float(v.median()),
            "p75": float(v.quantile(0.75)),
        }

    # --- CONFIDENCE ---------------------------------------------------
    # Three independent things have to hold for evidence to be strong:
    # enough distinct episodes, tight matches, and agreement among them.
    # The score is the product, so any one of them being poor drags it down.
    n_score = min(n_eff / 60.0, 1.0)
    # Tightness compares the selected analogues against the pool's MEDIAN
    # distance: 1.0 means they are far closer than a random episode, 0 means
    # they are no better than chance.
    med_all = float(np.median(d))
    tight = float(1.0 - np.mean(sel_d[close]) / med_all) if n_eff and med_all > 0 else 0.0
    tight = max(0.0, min(tight * 2.0, 1.0))
    tight = max(0.0, min(tight, 1.0))
    key = "label_tp_before_sl" if "label_tp_before_sl" in summary else (
        "label_touch" if "label_touch" in summary else None)
    if key and summary[key]["n"] > 1:
        pr = summary[key]["mean"]
        agree = abs(pr - 0.5) * 2.0          # 0 at a coin flip, 1 at certainty
    else:
        agree = 0.0
    confidence = float(n_score * tight * (0.5 + 0.5 * agree))

    out = {
        "symbol": symbol,
        "as_of": str(pd.Timestamp(q_time).date()),
        "scope": scope,
        "raw_neighbours": int(len(near)),
        "after_declustering": int(len(sel)),
        # NOT an effective sample size. Declustering removes same-symbol
        # temporal overlap only. It does nothing about market-wide or
        # sector-wide shocks, correlated names, or simultaneous regime
        # changes - 75 cross-stock analogues drawn from one NIFTY selloff are
        # nowhere near 75 independent observations. Named for what it is.
        "declustered_evidence_count": n_eff,
        "pit_epoch": epoch_used,
        "cutoff_session": str(pd.Timestamp(cutoff).date()),
        "horizon_sessions": horizon,
        "median_distance": float(np.median(sel_d[close])) if n_eff else None,
        "distance_cut": float(cut),
        "pool_median_distance": float(np.median(d)),
        "pool_size": int(len(pool)),
        "evidence_score": round(confidence, 3),
        "components": {"n": round(n_score, 3), "tightness": round(tight, 3),
                       "agreement": round(agree, 3)},
        "outcomes": summary,
        "analogues": [
            {"symbol": r["symbol"], "date": str(pd.Timestamp(r["timestamp"]).date()),
             "distance": round(float(dd), 3),
             "touch": (None if pd.isna(r.get("label_touch"))
                       else int(r.get("label_touch")))}
            for (_, r), dd in zip(rows.head(8).iterrows(), sel_d[close][:8])
        ],
    }
    if verbose:
        print(_render(out))
    return out


def _render(o: dict) -> str:
    if "error" in o:
        return f"{o['symbol']} - {o['error']}"
    L = [f"\n{o['symbol']} - as of {o['as_of']}  (scope: {o['scope']})", ""]
    L.append(f"  PIT normalisation       frozen {o['pit_epoch']}"
             f"   (no post-query data used)")
    L.append(f"  Leakage cutoff          {o['cutoff_session']}"
             f"   ({o['horizon_sessions']} sessions before query)")
    L.append(f"  Searched                {o['raw_neighbours']} nearest episodes")
    L.append(f"  After declustering      {o['after_declustering']}"
             f"   (near-duplicate days removed)")
    L.append(f"  Within distance cut     {o['declustered_evidence_count']}"
             f"   <- declustered, NOT independent observations")
    if o.get("median_distance") is not None:
        L.append(f"  Median distance         {o['median_distance']:.3f}"
                 f"   vs {o['pool_median_distance']:.3f} for a random episode")
        L.append(f"  Pool searched           {o['pool_size']:,} prior episodes")
    L.append("")
    lab = {
        "label_tp_before_sl": "P(TP before SL)",
        "label_touch": "P(+5% within 5d)",
        "label_fwd_ret_5d": "5d return",
        "label_mfe_5d": "MFE",
        "label_mae_5d": "MAE",
        "label_days_to_tp": "Days to target",
    }
    for c, s in o["outcomes"].items():
        nm = lab.get(c, c)
        if c.startswith("label_t"):
            L.append(f"  {nm:<22} {s['mean']:.1%}   (n={s['n']})")
        elif "days" in c:
            L.append(f"  {nm:<22} {s['median']:.1f}   "
                     f"(p25 {s['p25']:.0f} / p75 {s['p75']:.0f})")
        else:
            L.append(f"  {nm:<22} {s['mean']:+.2%}  "
                     f"(p25 {s['p25']:+.2%} / p75 {s['p75']:+.2%})")
    c = o["components"]
    L += ["", f"  Evidence score          {o['evidence_score']:.2f}",
          f"    episodes {c['n']:.2f} x tightness {c['tightness']:.2f} "
          f"x agreement {c['agreement']:.2f}", ""]
    if o["analogues"]:
        L.append("  Closest analogues:")
        for a in o["analogues"]:
            t = "" if a["touch"] is None else ("  +5% hit" if a["touch"] else "  no")
            L.append(f"    {a['symbol']:<14} {a['date']}   d={a['distance']:.3f}{t}")
    L += ["",
          "  EVIDENCE STRENGTH, NOT A CALIBRATED PROBABILITY. These are",
          "  historical outcomes from similar states; nothing here has been",
          "  checked against realised predictions. It earns the word",
          "  'confidence' only once the ledger shows it holding out-of-sample."]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["build", "ask"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--panel", default=None)
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--date", default=None)
    ap.add_argument("--scope", default="self", choices=["self", "cross", "both"])
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    a = ap.parse_args()

    root = Path(a.root or os.environ.get("CACHE_DAILY_ROOT", "."))
    pdir = Path(a.panel) if a.panel else root / "panel"
    ppq = pdir / "panel.parquet" if pdir.is_dir() else pdir

    if a.cmd == "build":
        build_memory(ppq)
    else:
        if not a.symbol:
            print("ask needs --symbol", file=sys.stderr)
            return 2
        if a.scope == "both":
            for sc in ("self", "cross"):
                ask(ppq, a.symbol, date=a.date, scope=sc, k=a.k)
        else:
            ask(ppq, a.symbol, date=a.date, scope=a.scope, k=a.k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
