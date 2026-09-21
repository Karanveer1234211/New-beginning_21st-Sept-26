#!/usr/bin/env python3
"""
Daily orchestration for the NSE quant cache.

    python run_daily.py --symbols-file watchlist.txt

Runs three stages and STOPS at the first failure, because a partial cache
feeding a panel is worse than no run at all:

    1. CACHE    fetch raw OHLCV for equities + exogenous series
    2. GATE     cross-checked calendar, absence classification, quarantine
    3. PANEL    your own build step (wire it in at run_panel())

Stage 2 is the point of the whole thing. It is not a report you skim; it is
the thing that decides whether stage 3 is allowed to happen.

SCHEDULE
--------
Run after 23:45 IST. MCX trades to 23:30, so an earlier run stores a PARTIAL
crude/gold candle. v24.1's REFETCH_TAIL_DAYS repairs that on the next run,
but a late run means the data is right the first time.

If you must run at ~16:00 for the equity signal, run stage 1 twice: once
after NSE close for the equity picks, once after MCX close to settle the
macro bars. Do not build the panel off the 16:00 pass.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "Daily_cache_v24.py"


def run_cache(symbols_file: str, extra: list[str]) -> None:
    cmd = [sys.executable, str(CACHE), "--symbols-file", symbols_file,
           "--incremental", *extra]
    print(f"[1/3] CACHE  {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"cache build failed (exit {r.returncode}); stopping")


def run_gate(root: Path, symbols: list[str], *, allow_partial: bool) -> list[str]:
    sys.path.insert(0, str(HERE))
    from data_quality import assert_panel_ready, DataQualityError

    print(f"[2/3] GATE   auditing {len(symbols)} series under {root}", flush=True)
    try:
        allowed = assert_panel_ready(root, symbols, allow_partial=allow_partial)
    except DataQualityError as e:
        print(e, file=sys.stderr)
        raise SystemExit(
            "\nQUARANTINE: not building a panel on this data.\n"
            "Inspect quarantine.json, then either fix the cache "
            "(--force on the affected symbols) or rerun with --allow-partial "
            "if a thinner universe is acceptable today."
        )
    print(f"       {len(allowed)}/{len(symbols)} cleared", flush=True)
    return allowed


def run_panel(root: Path, allowed: list[str]) -> None:
    """
    WIRE YOUR PANEL BUILD IN HERE.

    It should read ONLY `allowed`, and it should read each symbol through
    features_daily, never straight off the parquet:

        import features_daily as fx
        fx.assert_leak_free()                      # v20 gates, once per process
        for sym in allowed:
            raw  = pd.read_parquet(root / f"{sym}.parquet")
            feat = fx.build_features(raw)          # full history in, flagged out
            feat = fx.drop_warmup(feat)
            X    = feat[fx.feature_columns(feat)]  # NOT feat.columns - label!

    Cross-sectional features (true WQ ranks, sector-relative, beta) belong
    after this loop, once every symbol is stacked into one frame.
    """
    print("[3/3] PANEL  not wired up yet - see run_panel() in this file")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols-file", required=True)
    ap.add_argument("--root", default=None, help="cache dir (default: from env)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="build the panel from whatever passed the gate")
    ap.add_argument("--skip-cache", action="store_true",
                    help="gate and panel only, no fetch")
    args, extra = ap.parse_known_args()

    if not args.skip_cache:
        run_cache(args.symbols_file, extra)

    sys.path.insert(0, str(HERE))
    from Daily_cache_v24 import Config

    root = Path(args.root) if args.root else Config.from_env().day_root()
    from data_quality import discover_symbols
    symbols = discover_symbols(root)
    if not symbols:
        raise SystemExit(f"no cached symbols under {root}")

    allowed = run_gate(root, symbols, allow_partial=args.allow_partial)
    run_panel(root, allowed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
