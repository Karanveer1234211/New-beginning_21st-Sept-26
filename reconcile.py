#!/usr/bin/env python3
"""
reconcile.py - check the cache against an INDEPENDENT price source.

    python reconcile.py --root <cache_root> --sample 100
    python reconcile.py --root <cache_root> --symbols RELIANCE TCS --days 500

WHY THIS EXISTS
===============
Everything else in this pipeline verifies internal consistency: the maths is
right, the sessions are complete, nothing reads the future. None of it can
detect the cache faithfully storing what Kite sent when Kite was wrong.

A second vendor disagreeing does not prove Kite is wrong. It proves one of
them is, which is the whole point - right now you have no way to know.

WHAT IS COMPARED, AND WHY IT IS RETURNS
---------------------------------------
Adjusted price LEVELS are not comparable across vendors: each applies its own
split and dividend adjustments from its own reference date, so two correct
series can sit at different absolute levels forever. Daily RETURNS are
comparable - they are adjustment-invariant except on the adjustment date
itself.

So the primary test is close-to-close return agreement. A mismatch means one
of four things, in rough order of likelihood:

    1. one vendor missed or mis-dated a corporate action
    2. one vendor has a bad print on that day
    3. the series are the same name at different exchanges (NSE vs BSE)
    4. a genuine data error in the cache

Volume is compared separately and loosely: vendors legitimately differ on
what they count (delivery vs total, exchange coverage), so a persistent level
difference there is informative but not an error by itself.

SOURCES
-------
Default is yfinance (`SYMBOL.NS`). It is free, independent of Kite, and good
enough to catch the errors that matter. It is NOT a reference standard - on a
disagreement, check the exchange's own bhavcopy before assuming Kite is wrong.

Plug in another source by passing a fetcher: a callable
(symbol, start, end) -> DataFrame[timestamp, open, high, low, close, volume].
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data_quality import _paths, _read_ok, discover_symbols

# A day where returns differ by more than this is a disagreement worth a look.
RET_TOL = 0.005          # 50 bps
# Share of overlapping days that may disagree before the symbol is flagged.
MAX_MISMATCH_RATE = 0.01
# A return gap this large on a single day smells like a corporate action.
CA_SUSPECT = 0.15
# A single day this far apart is a failure on its own, whatever the average
# says. A rate threshold alone cannot see an isolated catastrophic error:
# one bad print in 500 sessions is a 0.4% mismatch rate - under any sane
# rate limit - while being exactly the kind of error that ruins a backtest.
SEVERE_DIFF = 0.05


def yfinance_fetcher(symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Independent source: Yahoo, NSE listing. Requires `pip install yfinance`."""
    import yfinance as yf

    t = yf.Ticker(f"{symbol}.NS")
    df = t.history(start=start, end=end + dt.timedelta(days=1),
                   auto_adjust=False, actions=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    df = df.rename(columns={"date": "timestamp"})
    keep = [c for c in ("timestamp", "open", "high", "low", "close", "volume")
            if c in df.columns]
    out = df[keep].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"]).dt.tz_localize(None).dt.normalize()
    return out


def _cached(root: Path, symbol: str) -> pd.DataFrame:
    pq, _ = _paths(root, symbol)
    if not pq.exists():
        return pd.DataFrame()
    df = pd.read_parquet(pq, columns=["timestamp", "open", "high", "low",
                                      "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None).dt.normalize()
    return df


def reconcile_symbol(
    root: Path,
    symbol: str,
    *,
    fetcher: Callable = yfinance_fetcher,
    days: int = 750,
    ret_tol: float = RET_TOL,
) -> dict:
    ours = _cached(root, symbol)
    if ours.empty:
        return {"symbol": symbol, "status": "NOT_CACHED"}

    end = ours["timestamp"].max().date()
    start = max(ours["timestamp"].min().date(), end - dt.timedelta(days=days))
    try:
        theirs = fetcher(symbol, start, end)
    except Exception as e:
        return {"symbol": symbol, "status": "SOURCE_ERROR", "error": repr(e)[:150]}
    if theirs is None or theirs.empty:
        return {"symbol": symbol, "status": "NO_REFERENCE_DATA"}

    ours = ours.loc[ours["timestamp"] >= pd.Timestamp(start)]
    a = ours.set_index("timestamp").sort_index()
    b = theirs.set_index("timestamp").sort_index()
    common = a.index.intersection(b.index)
    if len(common) < 30:
        return {"symbol": symbol, "status": "TOO_LITTLE_OVERLAP",
                "overlap_days": int(len(common))}

    # Sessions each has that the other does not. A session missing from OUR
    # cache is the serious direction.
    only_ours = a.index.difference(b.index)
    only_theirs = b.index.difference(a.index)

    ra = a.loc[common, "close"].pct_change()
    rb = b.loc[common, "close"].pct_change()
    diff = (ra - rb).abs()
    valid = diff.notna()
    n = int(valid.sum())
    mism = diff > ret_tol
    n_mism = int((mism & valid).sum())

    ca_like = diff[(diff > CA_SUSPECT) & valid]
    worst = diff[valid].sort_values(ascending=False).head(5)

    va = pd.to_numeric(a.loc[common, "volume"], errors="coerce")
    vb = pd.to_numeric(b.loc[common, "volume"], errors="coerce")
    with np.errstate(invalid="ignore", divide="ignore"):
        vratio = (va / vb.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    rate = (n_mism / n) if n else np.nan
    max_diff = float(diff[valid].max()) if n else np.nan

    # Two independent failure conditions: too many small disagreements, or
    # any single severe one.
    reasons = []
    if not n:
        reasons.append("no comparable returns")
    else:
        if rate > MAX_MISMATCH_RATE:
            reasons.append(f"mismatch rate {rate:.2%} > {MAX_MISMATCH_RATE:.0%}")
        if max_diff > SEVERE_DIFF:
            reasons.append(f"worst single day differs by {max_diff:.1%}")
    if len(only_theirs) > 3:
        reasons.append(f"{len(only_theirs)} sessions absent from the cache")

    if not reasons:
        status = "OK"
    elif len(only_theirs) > 3:
        status = "MISSING_SESSIONS"
    elif len(ca_like) and max_diff > SEVERE_DIFF and rate <= MAX_MISMATCH_RATE:
        status = "CORPORATE_ACTION"
    else:
        status = "MISMATCH"

    return {
        "symbol": symbol,
        "status": status,
        "overlap_days": int(len(common)),
        "compared_returns": n,
        "mismatches": n_mism,
        "mismatch_rate": round(float(rate), 5) if n else None,
        "max_return_diff": round(max_diff, 5) if n else None,
        "reasons": reasons,
        "corporate_action_suspects": [
            {"date": str(d.date()), "return_diff": round(float(v), 4)}
            for d, v in ca_like.items()
        ][:5],
        "worst_days": [
            {"date": str(d.date()), "return_diff": round(float(v), 5)}
            for d, v in worst.items()
        ],
        "sessions_only_in_cache": [str(d.date()) for d in only_ours[:5]],
        "sessions_missing_from_cache": [str(d.date()) for d in only_theirs[:5]],
        "n_sessions_missing_from_cache": int(len(only_theirs)),
        "median_volume_ratio": (round(float(vratio.median()), 3)
                                if vratio.notna().any() else None),
    }


def reconcile(
    root,
    symbols: Optional[Sequence[str]] = None,
    *,
    sample: int = 50,
    days: int = 750,
    fetcher: Callable = yfinance_fetcher,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    root = Path(root)
    if symbols:
        pick = list(symbols)
    else:
        eq = [s for s in discover_symbols(root)
              if (_read_ok(_paths(root, s)[1]) or {}).get("series_kind", "equity")
              == "equity"]
        random.Random(seed).shuffle(eq)
        pick = eq[:sample]

    results = []
    for i, s in enumerate(pick, 1):
        r = reconcile_symbol(root, s, fetcher=fetcher, days=days)
        results.append(r)
        if verbose:
            print(f"  [{i}/{len(pick)}] {s:<16} {r['status']:<18} "
                  f"rate={r.get('mismatch_rate')}")

    by_status: Dict[str, int] = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    checked = [r for r in results if r["status"] in ("OK", "MISMATCH",
                                                     "MISSING_SESSIONS")]
    rates = [r["mismatch_rate"] for r in checked if r.get("mismatch_rate") is not None]

    report = {
        "generated_at": dt.datetime.now().isoformat(),
        "root": str(root),
        "symbols_checked": len(pick),
        "by_status": by_status,
        "median_mismatch_rate": round(float(np.median(rates)), 5) if rates else None,
        "symbols_with_ca_suspects": [
            r["symbol"] for r in results if r.get("corporate_action_suspects")
        ],
        "results": results,
    }
    try:
        (root / "reconciliation.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--days", type=int, default=750)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rep = reconcile(a.root, a.symbols, sample=a.sample, days=a.days, seed=a.seed)
    print("\n" + "=" * 62)
    print("RECONCILIATION vs INDEPENDENT SOURCE")
    print("=" * 62)
    for k, v in sorted(rep["by_status"].items()):
        print(f"  {k:<20} {v}")
    print(f"  median mismatch rate: {rep['median_mismatch_rate']}")
    if rep["symbols_with_ca_suspects"]:
        print(f"\n  CORPORATE ACTION SUSPECTS "
              f"({len(rep['symbols_with_ca_suspects'])}): "
              f"{', '.join(rep['symbols_with_ca_suspects'][:12])}")
        print("  A single-day return gap >15% usually means one vendor applied "
              "a split the other did not.")
    bad = [r for r in rep["results"]
           if r["status"] in ("MISMATCH", "MISSING_SESSIONS", "CORPORATE_ACTION")]
    if bad:
        print(f"\n  NEEDS INVESTIGATION ({len(bad)}):")
        for r in bad[:15]:
            print(f"    {r['symbol']:<16} {r['status']:<18} "
                  f"{'; '.join(r.get('reasons') or [])[:60]}")
            if r.get("worst_days"):
                w = r["worst_days"][0]
                print(f"      worst: {w['date']} diff={w['return_diff']}")
    print(f"\n  JSON: {Path(a.root) / 'reconciliation.json'}")
    print("\n  NOTE: disagreement identifies a conflict, not a culprit. Check "
          "the exchange bhavcopy\n  before assuming the cache is the wrong one.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
