#!/usr/bin/env python3
"""
soul_features.py - turn the memory layer into MODEL FEATURES, point-in-time.

    python soul_features.py build --root <cache_root>
    python soul_features.py build --root <cache_root> --every 5
    python soul_features.py audit --root <cache_root>

WHAT THIS IS FOR
================
The memory engine answers one question at a time. To find out whether it
carries information the base model does not already have, every historical
row needs the same numbers the live card shows - computed as of that row's
own date. Then the walk-forward test answers the only question that matters:

    does adding these columns improve out-of-sample performance?

Soul does not decide anything here. It emits evidence. The model decides, and
the walk-forward decides whether the model was right to.

THE FEATURES, AND WHY THE GAPS MATTER MOST
------------------------------------------
Per row: self-memory block, cross-memory block, and the DISAGREEMENT between
them. The gaps are the interesting part. "This stock has never done this, but
the market has, often, and it went nowhere" is a different situation from
"both agree strongly" - and a single blended probability would erase that
distinction entirely.

TWO APPROXIMATIONS, STATED PLAINLY
==================================
1. THE CROSS POOL IS SUBSAMPLED.
   An exact search is ~10^12 distance computations. The cross pool is instead
   a deterministic, time-stratified sample (default 60k episodes) drawn only
   from dates at or before the cutoff. Deterministic so a rebuild reproduces
   itself; time-stratified so no era dominates. Consequence: these features
   will NOT exactly reproduce `memory.py ask`, which searches everything.
   They are the same statistic measured on a sample.

2. FEATURES ARE COMPUTED EVERY N SESSIONS AND HELD FORWARD.
   With --every 5, a row inherits the most recent computed value at or before
   its date. Holding a PAST value forward is point-in-time safe; it just means
   the feature is up to N-1 sessions stale. Set --every 1 for exactness at
   roughly N times the cost.

Both degrade precision, neither leaks. If the walk-forward shows these
features helping at --every 5, it is worth re-running at --every 1 to see
how much was lost to staleness.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import memory as M

CROSS_POOL = 60_000        # time-stratified sample of prior episodes
TOP_K = 100                # analogues kept per query
SELF_MIN_POOL = 60         # below this, self-memory emits nulls

FEATURE_COLUMNS = [
    # self block
    "soul_self_n", "soul_self_similarity", "soul_self_tp_first",
    "soul_self_touch", "soul_self_ret5", "soul_self_mfe", "soul_self_mae",
    "soul_self_strength",
    # cross block
    "soul_cross_n", "soul_cross_similarity", "soul_cross_tp_first",
    "soul_cross_touch", "soul_cross_ret5", "soul_cross_mfe", "soul_cross_mae",
    "soul_cross_strength",
    # the disagreement block - the part a blended probability would destroy
    "soul_gap_tp_first", "soul_gap_ret5", "soul_gap_mfe",
    "soul_agreement", "soul_dispersion", "soul_self_available",
]


QUERY_BATCH = 128          # query rows per distance block, to bound RAM


def _validate(every: int, cross_pool: int, top_k: int) -> None:
    """
    Fail now, not forty minutes into a build.

    A bad parameter here surfaces as an obscure numpy error deep in the loop,
    after the user has already waited. Cheap to check up front.
    """
    if every < 1:
        raise ValueError(f"--every must be >= 1, got {every}")
    if top_k < 1:
        raise ValueError(f"--top-k must be >= 1, got {top_k}")
    if cross_pool < top_k:
        raise ValueError(
            f"--cross-pool ({cross_pool}) must be >= --top-k ({top_k}): "
            f"cannot select more analogues than the pool contains")
    if SELF_MIN_POOL <= 0:
        raise ValueError(f"SELF_MIN_POOL must be > 0, got {SELF_MIN_POOL}")
    if cross_pool < 1000:
        raise ValueError(
            f"--cross-pool {cross_pool} is too small to be representative; "
            f"use at least 1000")


def _cdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Euclidean distances between every row of A and every row of B.

    Uses ||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b rather than broadcasting the
    difference. The broadcast form materialises an (nA, nB, d) array - at
    1500 queries x 60k pool x 12 dims that is ~8.6 GB, which is the single
    largest engineering risk in this file. The dot-product form allocates
    only (nA, nB).

    Clipped at zero because floating-point cancellation can make the
    expansion very slightly negative for near-identical rows, and sqrt of a
    negative is nan.
    """
    a2 = (A * A).sum(axis=1)[:, None]
    b2 = (B * B).sum(axis=1)[None, :]
    d2 = a2 + b2 - 2.0 * (A @ B.T)
    return np.sqrt(np.maximum(d2, 0.0))


def _stratified_pool(n_prior: int, cap: int) -> np.ndarray:
    """
    Deterministic, time-spread sample of the first `n_prior` episodes.

    Evenly spaced rather than random: the index is time-ordered, so a stride
    samples every era in proportion. A random draw would do the same in
    expectation but would not reproduce across rebuilds.
    """
    if n_prior <= cap:
        return np.arange(n_prior)
    stride = n_prior / float(cap)
    return np.unique((np.arange(cap) * stride).astype(np.int64))


def _summarise(rows: pd.DataFrame, d: np.ndarray, outs: List[str]) -> dict:
    """Outcome means over the selected analogues, plus dispersion."""
    o = {"n": int(len(rows)), "sim": float(np.mean(d)) if len(d) else np.nan}
    for c in outs:
        v = pd.to_numeric(rows[c], errors="coerce")
        o[c] = float(v.mean()) if v.notna().any() else np.nan
    key = "label_tp_before_sl"
    if key in rows.columns:
        v = pd.to_numeric(rows[key], errors="coerce").dropna()
        # Dispersion of a binary outcome: 0 when unanimous, 1 at a coin flip.
        o["disp"] = float(4.0 * v.mean() * (1 - v.mean())) if len(v) else np.nan
    else:
        o["disp"] = np.nan
    return o


def build(panel_path, *, every: int = 5, cross_pool: int = CROSS_POOL,
          top_k: int = TOP_K, verbose: bool = True) -> Path:
    panel_path = Path(panel_path)
    md = M.memory_dir(panel_path)
    if not (md / "state_raw.npy").exists():
        raise RuntimeError("memory not built - run `memory.py build` first")

    Xraw = np.load(md / "state_raw.npy")
    ez = np.load(md / "edges.npz", allow_pickle=False)
    all_edges, epochs = ez["edges"], ez["epochs"]
    idx = pd.read_parquet(md / "index.parquet")
    meta = json.loads((md / "memory_meta.json").read_text(encoding="utf-8"))
    horizon = int(meta.get("horizon_sessions", 5))
    outs = [c for c in meta["outcome_columns"] if c in idx.columns]

    ts = idx["timestamp"].to_numpy(dtype="datetime64[ns]")
    syms = idx["symbol"].to_numpy()
    sessions = np.unique(ts)
    q_dates = sessions[::max(1, every)]
    # A query needs `horizon` prior sessions to have a legal cutoff, and a
    # frozen normalisation to exist.
    q_dates = q_dates[q_dates >= epochs[0]]
    q_dates = q_dates[np.searchsorted(sessions, q_dates) >= horizon]

    if verbose:
        print(f"  {len(idx):,} episodes | {len(q_dates):,} query dates "
              f"(every {every} sessions) | cross pool {cross_pool:,}")

    _validate(every, cross_pool, top_k)

    by_sym: Dict[str, np.ndarray] = {}
    for s in np.unique(syms):
        by_sym[s] = np.where(syms == s)[0]

    recs: List[dict] = []
    t0 = time.perf_counter()
    for di, qd in enumerate(q_dates, 1):
        qpos = int(np.searchsorted(sessions, qd, side="right")) - 1
        cutoff = sessions[qpos - horizon]
        ei = int(np.where(epochs <= qd)[0][-1])
        edges = all_edges[ei]

        q_rows = np.where(ts == qd)[0]
        if not len(q_rows):
            continue
        prior = np.where(ts <= cutoff)[0]
        if len(prior) < 500:
            continue

        sub = prior[_stratified_pool(len(prior), cross_pool)]
        Xc = M._normalise(Xraw[sub], edges)
        Xq = M._normalise(Xraw[q_rows], edges)
        c_syms = syms[sub]

        # one distance matrix for every query on this date
        # Distances in query batches so peak RAM is (QUERY_BATCH x pool),
        # not (all queries x pool x dims).
        rnd_med = float(np.median(_cdist(Xq[:min(64, len(Xq))], Xc)))
        if not (rnd_med > 0):
            continue

        for b0 in range(0, len(q_rows), QUERY_BATCH):
          b1 = min(b0 + QUERY_BATCH, len(q_rows))
          D = _cdist(Xq[b0:b1], Xc)
          for qi_local, ridx in enumerate(q_rows[b0:b1]):
            qi = b0 + qi_local
            sym = syms[ridx]
            d_row = D[qi_local]
            # ---- cross: exclude the stock itself ----
            m = c_syms != sym
            dm = np.where(m, d_row, np.inf)
            k = min(top_k, int(m.sum()))
            sel = np.argpartition(dm, k - 1)[:k]
            sel = sel[np.argsort(dm[sel])]
            cross = _summarise(idx.iloc[sub[sel]], dm[sel], outs)

            # ---- self: that symbol's own prior episodes, declustered ----
            own = by_sym[sym]
            own = own[ts[own] <= cutoff]
            if len(own) >= SELF_MIN_POOL:
                Xs = M._normalise(Xraw[own], edges)
                ds = np.sqrt(((Xs - Xq[qi]) ** 2).sum(axis=1))
                o = np.argsort(ds)
                own_pos = M.session_positions(ts[own], sessions)
                keep = M._decluster(o, own_pos, syms[own], horizon)[:max(top_k // 4, 5)]
                self_ = _summarise(idx.iloc[own[keep]], ds[keep], outs)
                self_ok = 1.0
            else:
                self_ = {"n": 0, "sim": np.nan, "disp": np.nan,
                         **{c: np.nan for c in outs}}
                self_ok = 0.0

            def g(b, c):
                return b.get(c, np.nan)

            # Similarity is expressed RELATIVE to a random episode on this
            # date, so it means the same thing across dates and regimes.
            c_sim = 1.0 - cross["sim"] / rnd_med if rnd_med > 0 else np.nan
            s_sim = (1.0 - self_["sim"] / rnd_med
                     if self_ok and rnd_med > 0 else np.nan)

            rec = {
                "timestamp": qd, "symbol": sym,
                "soul_self_n": self_["n"],
                "soul_self_similarity": s_sim,
                "soul_self_tp_first": g(self_, "label_tp_before_sl"),
                "soul_self_touch": g(self_, "label_touch"),
                "soul_self_ret5": g(self_, "label_fwd_ret_5d"),
                "soul_self_mfe": g(self_, "label_mfe_5d"),
                "soul_self_mae": g(self_, "label_mae_5d"),
                # A HEURISTIC, not a statistical quantity - hence "strength"
                # rather than "evidence". Similarity is clipped at 0 here
                # because a negative value (analogues FARTHER than a random
                # episode) would flip the product's sign and read as strong
                # when it means the opposite. The raw, unclipped similarity
                # stays in its own column so the model still sees it.
                "soul_self_strength": (min(self_["n"] / 30.0, 1.0)
                                       * (max(s_sim, 0.0)
                                          if np.isfinite(s_sim) else 0.0)),
                "soul_cross_n": cross["n"],
                "soul_cross_similarity": c_sim,
                "soul_cross_tp_first": g(cross, "label_tp_before_sl"),
                "soul_cross_touch": g(cross, "label_touch"),
                "soul_cross_ret5": g(cross, "label_fwd_ret_5d"),
                "soul_cross_mfe": g(cross, "label_mfe_5d"),
                "soul_cross_mae": g(cross, "label_mae_5d"),
                "soul_cross_strength": (min(cross["n"] / 100.0, 1.0)
                                        * (max(c_sim, 0.0)
                                           if np.isfinite(c_sim) else 0.0)),
                "soul_dispersion": cross["disp"],
                "soul_self_available": self_ok,
            }
            rec["soul_gap_tp_first"] = (rec["soul_self_tp_first"]
                                        - rec["soul_cross_tp_first"])
            rec["soul_gap_ret5"] = rec["soul_self_ret5"] - rec["soul_cross_ret5"]
            rec["soul_gap_mfe"] = rec["soul_self_mfe"] - rec["soul_cross_mfe"]
            # 1 when self and cross tell the same story, 0 when opposed.
            gp = rec["soul_gap_tp_first"]
            rec["soul_agreement"] = (1.0 - min(abs(gp), 1.0)
                                     if np.isfinite(gp) else np.nan)
            recs.append(rec)

        if verbose and di % 25 == 0:
            el = time.perf_counter() - t0
            eta = el / di * (len(q_dates) - di)
            print(f"    {di}/{len(q_dates)} dates | {len(recs):,} rows | "
                  f"eta {eta/60:.1f} min")

    if not recs:
        raise RuntimeError("no rows produced")
    out = pd.DataFrame(recs)
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    sd = panel_path.parent / "soul"
    sd.mkdir(parents=True, exist_ok=True)
    out.to_parquet(sd / "soul_features.parquet", index=False)
    (sd / "soul_features_meta.json").write_text(json.dumps({
        "built_at": dt.datetime.now().isoformat(),
        "rows": int(len(out)), "query_dates": int(len(q_dates)),
        "every_n_sessions": every, "cross_pool": cross_pool, "top_k": top_k,
        "features": FEATURE_COLUMNS,
        "approximations": [
            "cross pool is a deterministic time-stratified subsample",
            f"features computed every {every} sessions, held forward",
        ],
    }, indent=2), encoding="utf-8")
    if verbose:
        print(f"  {len(out):,} feature rows in "
              f"{(time.perf_counter()-t0)/60:.1f} min -> {sd}")
    return sd / "soul_features.parquet"


def attach(panel_path, feat_path=None) -> pd.DataFrame:
    """
    Join soul features onto the panel, holding each value forward.

    merge_asof with direction="backward" takes the most recent value at or
    before each row's date. Forward is never consulted, so a row can only
    ever see a feature computed from its own past.
    """
    panel_path = Path(panel_path)
    feat_path = Path(feat_path or panel_path.parent / "soul" / "soul_features.parquet")
    p = pd.read_parquet(panel_path)
    f = pd.read_parquet(feat_path)
    # merge_asof refuses mismatched datetime resolutions, and parquet
    # round-trips can differ (us vs ns). Normalise both to ns explicitly.
    p["timestamp"] = (pd.to_datetime(p["timestamp"]).dt.tz_localize(None)
                      .astype("datetime64[ns]"))
    f["timestamp"] = (pd.to_datetime(f["timestamp"]).dt.tz_localize(None)
                      .astype("datetime64[ns]"))
    p = p.sort_values(["timestamp", "symbol"])
    f = f.sort_values(["timestamp", "symbol"])
    return pd.merge_asof(p, f, on="timestamp", by="symbol",
                         direction="backward")


def audit(panel_path) -> None:
    """
    Integrity checks on the feature table. Each is a way the build could have
    gone wrong silently, so each FAILS LOUDLY rather than printing a warning
    nobody reads.
    """
    pp = Path(panel_path)
    sd = pp.parent / "soul"
    f = pd.read_parquet(sd / "soul_features.parquet")
    f["timestamp"] = pd.to_datetime(f["timestamp"])
    problems: List[str] = []

    print(f"\n  {len(f):,} rows | {f['symbol'].nunique()} symbols | "
          f"{f['timestamp'].min().date()} -> {f['timestamp'].max().date()}")

    # 1. duplicate keys would silently multiply rows on join
    dup = int(f.duplicated(["timestamp", "symbol"]).sum())
    print(f"  duplicate (timestamp, symbol) rows: {dup}")
    if dup:
        problems.append(f"{dup} duplicate keys")

    # 2. a feature dated after the panel ends means a calendar bug
    pts = pd.to_datetime(pd.read_parquet(pp, columns=["timestamp"])["timestamp"])
    pmax = pts.max().tz_localize(None) if pts.dt.tz else pts.max()
    beyond = int((f["timestamp"] > pmax).sum())
    print(f"  feature rows dated after the panel ends: {beyond}")
    if beyond:
        problems.append(f"{beyond} rows past panel end {pmax.date()}")

    # 3. self features must be absent exactly when self is unavailable
    na = f[f["soul_self_available"] == 0]
    leak = int(na[["soul_self_tp_first", "soul_self_ret5",
                   "soul_self_mfe"]].notna().any(axis=1).sum())
    print(f"  self features present despite self_available=0: {leak}")
    if leak:
        problems.append(f"{leak} rows carry self features they should not")

    # 4. cross features should never be missing - the pool is always there
    miss = int(f["soul_cross_tp_first"].isna().sum())
    print(f"  cross features missing: {miss}")
    if miss:
        problems.append(f"{miss} rows lack cross features")

    # 5. similarity is 1 - d/median: >1 is impossible, <0 means the
    #    analogues were worse than random
    for c in ("soul_self_similarity", "soul_cross_similarity"):
        v = pd.to_numeric(f[c], errors="coerce").dropna()
        hi = int((v > 1.0).sum()); lo = int((v < -1.0).sum())
        print(f"  {c}: min {v.min():.3f} max {v.max():.3f} "
              f"(>1: {hi}, <-1: {lo})")
        if hi:
            problems.append(f"{c} exceeds 1.0 on {hi} rows")

    # 6. agreement is a bounded score by construction
    ag = pd.to_numeric(f["soul_agreement"], errors="coerce").dropna()
    oob = int(((ag < 0) | (ag > 1)).sum())
    print(f"  soul_agreement outside [0,1]: {oob}")
    if oob:
        problems.append(f"agreement out of range on {oob} rows")

    # 7. attach must preserve the panel's row count exactly
    n_panel = len(pts)
    j = attach(pp)
    print(f"  attach: panel {n_panel:,} -> joined {len(j):,} "
          f"({j['soul_cross_tp_first'].notna().mean():.1%} covered)")
    if len(j) != n_panel:
        problems.append(f"attach changed row count {n_panel} -> {len(j)}")

    print(f"\n  self-memory available on {f['soul_self_available'].mean():.1%} "
          f"of rows (needs {SELF_MIN_POOL}+ prior episodes)\n")
    print(f"  {'feature':<28}{'non-null':>10}{'mean':>10}{'std':>10}")
    for c in FEATURE_COLUMNS:
        if c not in f.columns:
            continue
        v = pd.to_numeric(f[c], errors="coerce")
        print(f"  {c:<28}{v.notna().mean():>9.1%}{v.mean():>10.3f}"
              f"{v.std():>10.3f}")

    g = pd.to_numeric(f["soul_gap_tp_first"], errors="coerce").dropna()
    if len(g):
        print(f"\n  self-vs-cross TP gap: {(g > 0.1).mean():.1%} self more "
              f"bullish | {(g < -0.1).mean():.1%} cross more bullish | "
              f"{(g.abs() <= 0.1).mean():.1%} agree")

    if problems:
        print("\n  AUDIT FAILED:")
        for p in problems:
            print(f"    - {p}")
        raise SystemExit(1)
    print("\n  All integrity checks passed.")
    print("  NOTHING HERE IS VALIDATED. These are candidate features; the")
    print("  walk-forward decides whether they carry OOS information.\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["build", "audit"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--panel", default=None)
    ap.add_argument("--every", type=int, default=5,
                    help="compute every N sessions; 1 is exact and N times slower")
    ap.add_argument("--cross-pool", type=int, default=CROSS_POOL)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    a = ap.parse_args()

    root = Path(a.root or os.environ.get("CACHE_DAILY_ROOT", "."))
    pdir = Path(a.panel) if a.panel else root / "panel"
    ppq = pdir / "panel.parquet" if pdir.is_dir() else pdir

    if a.cmd == "build":
        build(ppq, every=a.every, cross_pool=a.cross_pool, top_k=a.top_k)
    else:
        audit(ppq)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
