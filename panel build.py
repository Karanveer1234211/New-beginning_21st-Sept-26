#!/usr/bin/env python3
"""
panel_build.py - assemble the model-ready panel.

    python panel_build.py --root <cache_root> --panel <panel_dir>
    python panel_build.py --root <cache_root> --panel <panel_dir> --full

ORDER, AND WHY IT IS THIS ORDER
===============================
    cache -> GATE -> per-symbol features -> point-in-time cross-section
          -> cross-sectional features -> exogenous join -> labels -> panel

The gate comes BEFORE the cross-section, not after. A quarantined symbol left
in a cross-sectional rank contaminates every other symbol's rank on that date,
not just its own row. Filtering afterwards does not undo that.

THE THREE THINGS THAT MAKE THIS MORE THAN AN APPEND
---------------------------------------------------
1. Per-symbol features are recomputed over FULL history every run. Expanding
   ranks depend on the whole series, and the cache's tail refetch revises
   recent bars, so recent feature values genuinely change. We recompute
   everything and rewrite only the recent window.

2. Labels arrive late. A 5-day-forward label for today is not knowable today.
   Tonight, rows from 5 sessions ago become resolvable. That is a backfill
   pass, not leakage - but it means the recent edge of the panel is always
   unlabelled, and training must treat those rows as UNKNOWN rather than as
   negatives.

3. The universe is point-in-time. Cross-sectional ranks for 2019 rank against
   the symbols that actually traded in 2019. Using today's symbol list is
   survivorship bias, and it is the easiest way to build a backtest that
   cannot be traded.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
Sector-relative features. They need dated index membership, which the cache
does not have. A sector map scraped today and applied to 2019 is both
lookahead and survivorship, in the feature family most likely to carry real
cross-sectional alpha. Pass --sector-map with a dated file to enable them.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import features_daily as fx
from data_quality import (
    assert_panel_ready, discover_symbols, _paths, _read_ok, build_calendar,
)

# Columns ranked cross-sectionally each date. These are the ones where
# "big relative to its peers today" is the actual hypothesis; ranking
# everything would just add 160 collinear columns.
CROSS_SECTIONAL_BASE = [
    "D_ret_1d_pct", "D_range_pct", "D_atr_pct", "D_rsi14", "D_adx14",
    "D_dist_from_52wh", "D_dist_from_52wl", "D_pos_in_52w_range",
    "D_drawdown_252", "D_dvol_z20", "D_dvol_z252", "D_realvol_20",
    "D_realvol_ratio_20_60", "D_amihud_20", "D_gap_pct",
    "D_intraday_ret_pct", "D_ret_skew_60", "D_downside_dev_60",
    "D_bb_pctB_20", "D_donch_pos_20", "D_ema20_angle_deg",
]

MARKET_SERIES = "NIFTY50"
LABEL_HORIZON = 5
LABEL_TOUCH_PCT = 0.05

PANEL_LABELS = ("label_touch", "label_fwd_ret_5d", "label_mfe_5d", "label_mae_5d")

# ----------------------------------------------------------------------
# WHICH UNIVERSE DOES THE PANEL REPRESENT?
#
# Two things are easy to conflate, so this states the rule explicitly:
#
#   TRADABILITY is point-in-time. A stock that had not listed in 2019 has no
#   2019 rows and cannot influence anyone's 2019 rank. That is handled by
#   construction - no bar, no row.
#
#   DATA QUALITY is a property of the WHOLE series. The gate inspects a
#   symbol's entire history: interior session gaps, OHLC violations, an
#   unresolved split. When a symbol is quarantined, the claim is not "it was
#   untradeable last week" - it is "this series' data cannot be trusted,
#   including its past".
#
# So a quarantined symbol has ALL its rows removed, history included. Keeping
# 2019 rows for a series we currently refuse to trust is the worst of both
# worlds: the panel would carry data the gate has rejected.
#
# The consequence is the important part. Cross-sectional ranks are computed
# ACROSS the universe on each date, so removing a symbol changes every other
# symbol's rank on every date it appeared. An incremental run cannot repair
# that by touching the recent tail. Therefore: if the allowed universe has
# changed since the last build, the panel is rebuilt in FULL, automatically.
# ----------------------------------------------------------------------


def _universe_hash(symbols: Sequence[str]) -> str:
    import hashlib
    return hashlib.sha256("|".join(sorted(symbols)).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------
# labels
# ----------------------------------------------------------------------
def compute_labels(
    df: pd.DataFrame, horizon: int = LABEL_HORIZON, touch: float = LABEL_TOUCH_PCT
) -> pd.DataFrame:
    """
    Forward outcomes for one symbol. DELIBERATELY forward-looking.

    label_touch is the BigMove target: did the high reach +touch% at any point
    in the next `horizon` sessions, measured from today's close.

    Every label is NaN where the window is incomplete. That matters: an
    unresolved row is UNKNOWN, not a negative. Filling it with 0 would teach
    the model that the most recent week never moves.
    """
    c = pd.to_numeric(df["close"], errors="coerce")
    h = pd.to_numeric(df["high"], errors="coerce")
    l = pd.to_numeric(df["low"], errors="coerce")

    # Forward windows exclude today's bar: shift(-1) then roll forward.
    fwd_max = h.shift(-1).rolling(horizon, min_periods=horizon).max().shift(-(horizon - 1))
    fwd_min = l.shift(-1).rolling(horizon, min_periods=horizon).min().shift(-(horizon - 1))
    fwd_close = c.shift(-horizon)

    out = pd.DataFrame(index=df.index)
    out["label_mfe_5d"] = (fwd_max / c) - 1.0
    out["label_mae_5d"] = (fwd_min / c) - 1.0
    out["label_fwd_ret_5d"] = (fwd_close / c) - 1.0
    reached = out["label_mfe_5d"] >= touch
    out["label_touch"] = reached.astype("Int8").mask(out["label_mfe_5d"].isna())
    return out


# ----------------------------------------------------------------------
# per-symbol stage
# ----------------------------------------------------------------------
def symbol_frame(root: Path, symbol: str, since: Optional[pd.Timestamp]) -> pd.DataFrame:
    """Full-history features for one symbol, sliced to the rows we will write."""
    pq, _ = _paths(root, symbol)
    raw = pd.read_parquet(pq)
    if raw.empty or len(raw) < fx.WARMUP_BARS + LABEL_HORIZON:
        return pd.DataFrame()

    feat = fx.build_features(raw)          # full history: expanding ranks need it

    # Labels are computed on the RAW frame and joined ON TIMESTAMP.
    #
    # Joining on the positional index is wrong and silently so: drop_warmup()
    # calls reset_index(), so after it the row numbers no longer correspond to
    # the raw frame. An index join would have paired each feature row with a
    # label from ~312 bars away - a model that trains beautifully and means
    # nothing. Timestamp is the only key that survives reindexing.
    lab = compute_labels(raw)
    lab.insert(0, "timestamp", pd.to_datetime(raw["timestamp"]).to_numpy())

    feat = fx.drop_warmup(feat)
    if feat.empty:
        return pd.DataFrame()
    feat["timestamp"] = pd.to_datetime(feat["timestamp"])
    if getattr(feat["timestamp"].dt, "tz", None) is not None:
        feat["timestamp"] = feat["timestamp"].dt.tz_localize(None)
    lab["timestamp"] = pd.to_datetime(lab["timestamp"])
    if getattr(lab["timestamp"].dt, "tz", None) is not None:
        lab["timestamp"] = lab["timestamp"].dt.tz_localize(None)

    before = len(feat)
    feat = feat.merge(lab, on="timestamp", how="left", validate="one_to_one")
    if len(feat) != before:
        raise RuntimeError(f"{symbol}: label join changed row count")

    feat.insert(1, "symbol", symbol)
    if since is not None:
        feat = feat.loc[feat["timestamp"] >= since]
    return feat.reset_index(drop=True)


# ----------------------------------------------------------------------
# cross-sectional stage
# ----------------------------------------------------------------------
def add_cross_sectional(panel: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """
    Rank within each date, across the symbols present ON that date.

    This is the real cross-sectional rank the per-symbol expanding rank was
    only ever standing in for. The universe per date is whoever has a row,
    which is point-in-time by construction: a symbol that had not listed has
    no row and cannot influence anyone else's rank.
    """
    have = [c for c in cols if c in panel.columns]
    if not have:
        return panel
    g = panel.groupby("timestamp", sort=False)
    ranked = {}
    for c in have:
        ranked[f"X_rank_{c}"] = g[c].rank(pct=True, method="average")
        z = (panel[c] - g[c].transform("mean")) / g[c].transform("std").replace(0, np.nan)
        ranked[f"X_z_{c}"] = z.clip(-5, 5)
    ranked["X_universe_size"] = g["symbol"].transform("size").astype("int32")
    return pd.concat([panel, pd.DataFrame(ranked, index=panel.index)], axis=1)


def add_market_relative(panel: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Return and volatility relative to the index, plus the index's own state."""
    pq, _ = _paths(root, MARKET_SERIES)
    if not pq.exists():
        print(f"  WARNING: {MARKET_SERIES} not cached; skipping market-relative")
        return panel
    mkt = pd.read_parquet(pq)
    mf = fx.drop_warmup(fx.build_features(mkt))
    keep = [c for c in ("D_ret_1d_pct", "D_realvol_20", "D_rsi14", "D_adx14",
                        "D_dist_from_52wh", "D_drawdown_252") if c in mf.columns]
    mf = mf[["timestamp"] + keep].copy()
    mf["timestamp"] = pd.to_datetime(mf["timestamp"]).dt.tz_localize(None)
    mf = mf.rename(columns={c: f"MKT_{c}" for c in keep})

    panel["timestamp"] = pd.to_datetime(panel["timestamp"]).dt.tz_localize(None)
    out = panel.merge(mf, on="timestamp", how="left")
    if "D_ret_1d_pct" in out.columns and "MKT_D_ret_1d_pct" in out.columns:
        out["X_excess_ret_1d"] = out["D_ret_1d_pct"] - out["MKT_D_ret_1d_pct"]
    if "D_realvol_20" in out.columns and "MKT_D_realvol_20" in out.columns:
        out["X_relvol_20"] = out["D_realvol_20"] / out["MKT_D_realvol_20"].replace(0, np.nan)
    return out


def add_exogenous(panel: pd.DataFrame, root: Path, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Join macro series onto the panel, already lagged by the cache's rules.

    Each exogenous series carries lag_sessions, set to 1 for anything closing
    after NSE. That lag is applied here, so a crude close that printed at
    23:30 can never appear in a same-day feature row.
    """
    import Daily_cache_v26 as dc
    added = []
    for s in dc.DEFAULT_EXOGENOUS:
        if s.name == MARKET_SERIES:
            continue
        pq, _ = _paths(root, s.name)
        if not pq.exists():
            continue
        raw = pd.read_parquet(pq)
        if raw.empty:
            continue
        al = dc.align_to_sessions(raw, sessions, series=s)
        if al.empty:
            continue
        keep = [c for c in al.columns if c.endswith(("_close", "_stale"))]
        al = al[keep].copy()
        al.index = pd.DatetimeIndex(al.index).tz_localize(None)
        al[f"{s.name}_ret_1d"] = pd.to_numeric(
            al.get(f"{s.name}_close"), errors="coerce"
        ).pct_change()
        al = al.reset_index().rename(columns={"index": "timestamp"})
        panel = panel.merge(al, on="timestamp", how="left")
        added.append(s.name)
    if added:
        print(f"  exogenous joined ({len(added)}): {', '.join(added[:8])}"
              + (" ..." if len(added) > 8 else ""))
    return panel


# ----------------------------------------------------------------------
# walk-forward with embargo
# ----------------------------------------------------------------------
def walk_forward_splits(
    panel: pd.DataFrame, n_splits: int = 5, embargo: int = LABEL_HORIZON
) -> List[dict]:
    """
    Expanding-window splits with an embargo between train and test.

    The embargo is not optional. A row dated t carries a label that resolves
    at t+5, so training on t and testing at t+1 lets the model see an outcome
    that overlaps the test period. `embargo` must be >= the label horizon.
    """
    if embargo < LABEL_HORIZON:
        raise ValueError(
            f"embargo {embargo} < label horizon {LABEL_HORIZON}: train labels "
            f"would overlap the test window"
        )
    dates = np.sort(pd.to_datetime(panel["timestamp"]).unique())
    if len(dates) < (n_splits + 1) * (embargo + 10):
        raise ValueError("not enough sessions for the requested splits")
    folds = np.array_split(dates, n_splits + 1)
    out = []
    for k in range(1, n_splits + 1):
        train_end = folds[k - 1][-1]
        test = folds[k]
        cut = pd.Timestamp(train_end) - pd.Timedelta(days=0)
        train_last = pd.Timestamp(dates[max(0, np.searchsorted(dates, cut) - embargo)])
        out.append({
            "fold": k,
            "train_start": str(pd.Timestamp(dates[0]).date()),
            "train_end": str(train_last.date()),
            "embargo_sessions": embargo,
            "test_start": str(pd.Timestamp(test[0]).date()),
            "test_end": str(pd.Timestamp(test[-1]).date()),
        })
    return out


# ----------------------------------------------------------------------
# main build
# ----------------------------------------------------------------------
def build_panel(
    root,
    panel_dir,
    *,
    symbols: Optional[Sequence[str]] = None,
    full: bool = False,
    rebuild_days: int = 15,
    allow_partial: bool = False,
    verbose: bool = True,
) -> Path:
    root, panel_dir = Path(root), Path(panel_dir)
    panel_dir.mkdir(parents=True, exist_ok=True)
    out_pq = panel_dir / "panel.parquet"

    syms = list(symbols) if symbols else discover_symbols(root)
    if verbose:
        print(f"[panel] gating {len(syms)} cached series")
    allowed = assert_panel_ready(root, syms, allow_partial=allow_partial)
    if not allowed:
        raise RuntimeError("no series cleared the data-quality gate")

    # Equities only in the cross-section; indices and futures are context.
    equities = []
    for s in allowed:
        meta = _read_ok(_paths(root, s)[1]) or {}
        if meta.get("series_kind", "equity") == "equity":
            equities.append(s)
    if verbose:
        print(f"[panel] {len(equities)} equities cleared, "
              f"{len(allowed) - len(equities)} context series")

    cal = build_calendar(root, allowed)
    sessions = cal.sessions
    if not len(sessions):
        raise RuntimeError("no trading calendar; cannot build a panel")

    uni_hash = _universe_hash(equities)
    prior_meta = {}
    meta_path = panel_dir / "panel_meta.json"
    if meta_path.exists():
        try:
            prior_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            prior_meta = {}

    if (not full) and prior_meta and prior_meta.get("universe_hash") != uni_hash:
        prior_n = prior_meta.get("symbols", "?")
        print(f"  UNIVERSE CHANGED ({prior_n} -> {len(equities)} symbols). "
              f"Cross-sectional ranks depend on who is in the universe on each "
              f"date, so a tail rewrite cannot repair history. Forcing a FULL "
              f"rebuild.")
        full = True

    existing = None
    since = None
    if not full and out_pq.exists():
        existing = pd.read_parquet(out_pq)
        if not existing.empty:
            # Rewrite the recent window: features there may have been revised
            # by the cache's tail refetch, and labels there have just become
            # resolvable.
            span = rebuild_days + LABEL_HORIZON
            cutoff_idx = max(0, len(sessions) - span)
            since = pd.Timestamp(sessions[cutoff_idx]).tz_localize(None)
            if verbose:
                print(f"[panel] incremental: rewriting from {since.date()} "
                      f"({span} sessions)")
    if full and verbose:
        print("[panel] FULL rebuild")

    frames, skipped = [], []
    for i, s in enumerate(equities, 1):
        try:
            f = symbol_frame(root, s, since)
            if not f.empty:
                frames.append(f)
        except Exception as e:
            skipped.append(f"{s}: {type(e).__name__}: {e}")
        if verbose and i % 200 == 0:
            print(f"  ...{i}/{len(equities)}")
    if skipped:
        print(f"  WARNING: {len(skipped)} symbol(s) failed: {skipped[:3]}")
    if not frames:
        raise RuntimeError("no symbol produced rows")

    panel = pd.concat(frames, ignore_index=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"]).dt.tz_localize(None)
    if verbose:
        print(f"[panel] stacked {len(panel):,} rows x {panel.shape[1]} cols")

    panel = add_cross_sectional(panel, CROSS_SECTIONAL_BASE)
    panel = add_market_relative(panel, root)
    panel = add_exogenous(panel, root, sessions)

    # Drop dates whose cross-section is too thin to rank meaningfully.
    thin = panel["X_universe_size"] < 20
    if thin.any():
        print(f"  dropping {int(thin.sum()):,} rows on dates with <20 symbols")
        panel = panel.loc[~thin]

    panel = panel.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    if existing is not None and since is not None:
        keep = existing.loc[
            pd.to_datetime(existing["timestamp"]).dt.tz_localize(None) < since
        ]
        # Belt and braces: even though a universe change forces a full rebuild
        # above, never carry forward rows for a symbol the gate does not
        # currently allow.
        allowed_set = set(equities)
        stale = ~keep["symbol"].isin(allowed_set)
        if stale.any():
            print(f"  dropping {int(stale.sum()):,} historical rows for "
                  f"{keep.loc[stale, 'symbol'].nunique()} now-quarantined symbol(s)")
            keep = keep.loc[~stale]
        panel = pd.concat([keep, panel], ignore_index=True)
        panel = panel.drop_duplicates(["timestamp", "symbol"], keep="last")
        panel = panel.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    panel.to_parquet(out_pq, index=False)

    lab = panel["label_touch"] if "label_touch" in panel.columns else pd.Series(dtype="Int8")
    resolved = int(lab.notna().sum())
    meta = {
        "built_at": dt.datetime.now().isoformat(),
        "rows": int(len(panel)),
        "columns": int(panel.shape[1]),
        "symbols": int(panel["symbol"].nunique()),
        "first_session": str(panel["timestamp"].min().date()),
        "last_session": str(panel["timestamp"].max().date()),
        "labels_resolved": resolved,
        "labels_pending": int(len(panel) - resolved),
        "label_positive_rate": float(lab.dropna().mean()) if resolved else None,
        "calendar_trusted": bool(cal.trusted),
        "universe_hash": uni_hash,
        "universe": sorted(equities),
        "feature_columns": len(panel_feature_columns(panel)),
        "mode": "full" if full else "incremental",
    }
    (panel_dir / "panel_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if verbose:
        print(f"[panel] {meta['rows']:,} rows | {meta['symbols']} symbols | "
              f"{meta['first_session']} -> {meta['last_session']}")
        print(f"[panel] labels resolved {resolved:,}, pending "
              f"{meta['labels_pending']:,} (the most recent "
              f"{LABEL_HORIZON} sessions are UNKNOWN, not negative)")
        if meta["label_positive_rate"] is not None:
            print(f"[panel] base rate: {meta['label_positive_rate']:.3%}")
        print(f"[panel] -> {out_pq}")
    return out_pq


def panel_feature_columns(panel: pd.DataFrame) -> List[str]:
    """
    Columns safe to hand a model, at panel level.

    Extends features_daily's registry with the panel's own labels. Build X
    with this; never with panel.columns.
    """
    banned = set(fx.non_feature_columns(panel)) | set(PANEL_LABELS) | {"symbol"}
    banned |= {c for c in panel.columns if c.startswith("label_")}
    return [c for c in panel.columns if c not in banned]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--full", action="store_true", help="rebuild all history")
    ap.add_argument("--rebuild-days", type=int, default=15)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--splits", type=int, default=0,
                    help="print N walk-forward folds after building")
    a = ap.parse_args()

    p = build_panel(a.root, a.panel, full=a.full, rebuild_days=a.rebuild_days,
                    allow_partial=a.allow_partial)
    if a.splits:
        panel = pd.read_parquet(p, columns=["timestamp"])
        for f in walk_forward_splits(panel, n_splits=a.splits):
            print(f"  fold {f['fold']}: train -> {f['train_end']} "
                  f"| embargo {f['embargo_sessions']} | "
                  f"test {f['test_start']}..{f['test_end']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
