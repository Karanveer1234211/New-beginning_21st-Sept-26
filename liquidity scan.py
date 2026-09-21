#!/usr/bin/env python3
"""
liquidity_scan.py - choose the turnover floor from YOUR data, not a default.

    python liquidity_scan.py --root <cache_root>
    python liquidity_scan.py --root <cache_root> --as-of 2019-06-28

Prints, for a range of candidate floors, how many symbols survive and what
share of the gappy names get removed. Run it before setting --min-turnover on
panel_build.

WHY --as-of MATTERS
-------------------
The universe is not fixed. Running this as of today tells you what is liquid
now; running it as of 2019 tells you what was liquid then, which is what your
2019 backtest will actually contain. If the two look very different, that is
the survivorship bias you would have baked in by editing watchlist.txt.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data_quality import _paths, _read_ok, discover_symbols

WINDOW = 60
CANDIDATES = [0, 1e5, 5e5, 1e6, 2.5e6, 5e6, 1e7, 2.5e7, 5e7, 1e8]


def symbol_turnover(root: Path, sym: str, as_of: Optional[pd.Timestamp]):
    """Median daily turnover over the WINDOW sessions ending at as_of."""
    pq, _ = _paths(root, sym)
    if not pq.exists():
        return None
    try:
        d = pd.read_parquet(pq, columns=["timestamp", "close", "volume"])
    except Exception:
        return None
    if d.empty:
        return None
    d["timestamp"] = pd.to_datetime(d["timestamp"]).dt.tz_localize(None)
    if as_of is not None:
        d = d.loc[d["timestamp"] <= as_of]
    if len(d) < WINDOW // 2:
        return None
    tail = d.tail(WINDOW)
    tv = pd.to_numeric(tail["close"], errors="coerce") * pd.to_numeric(
        tail["volume"], errors="coerce")
    tv = tv.dropna()
    if tv.empty:
        return None
    return {
        "symbol": sym,
        "median_turnover": float(tv.median()),
        "sessions": int(len(d)),
        "last": d["timestamp"].max().date(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--as-of", default=None,
                    help="measure liquidity as of this date (YYYY-MM-DD)")
    a = ap.parse_args()

    root = Path(a.root)
    as_of = pd.Timestamp(a.as_of) if a.as_of else None

    syms = [s for s in discover_symbols(root)
            if (_read_ok(_paths(root, s)[1]) or {}).get("series_kind", "equity")
            == "equity"]
    rows = [r for r in (symbol_turnover(root, s, as_of) for s in syms) if r]
    if not rows:
        print("no symbols with usable turnover")
        return 1
    df = pd.DataFrame(rows)

    label = f"as of {as_of.date()}" if as_of is not None else "as of latest bar"
    print(f"\nMedian daily turnover, {WINDOW}-session window, {label}")
    print(f"{len(df)} equities measured\n")

    q = df["median_turnover"].quantile([.05, .1, .25, .5, .75, .9, .95])
    print("  distribution (rupees/day):")
    for k, v in q.items():
        print(f"    {int(k*100):>3}th pct  {v:>18,.0f}")

    print(f"\n  {'floor (Rs/day)':>18} {'symbols':>9} {'share':>8}")
    for c in CANDIDATES:
        n = int((df["median_turnover"] >= c).sum())
        print(f"  {c:>18,.0f} {n:>9} {n/len(df):>7.1%}")

    print("\n  thinnest 10 by turnover:")
    for _, r in df.nsmallest(10, "median_turnover").iterrows():
        print(f"    {r['symbol']:<16} {r['median_turnover']:>16,.0f}  "
              f"last={r['last']}")

    print("\n  Set the floor from your OWN sizing: if a top-3 pick takes a")
    print("  position you cannot exit inside a few percent of daily turnover,")
    print("  the model's edge on that name is not reachable.")
    print("  Then: panel_build.py --min-turnover <value>")
    print("\n  Re-run with --as-of 2019-06-28 and compare. A large difference")
    print("  is the survivorship bias you would bake in by editing the")
    print("  watchlist instead of filtering per date.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
