"""Run an FL experiment grid (resumable: finished cells are skipped, failed ones are re-run).

    python scripts/fl/run_matrix.py --grid configs/fl/matrix_p0_defense.yaml --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from graphslm_ids.offline.federated.matrix import run_grid  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an FL experiment grid.")
    parser.add_argument("--grid", required=True)
    parser.add_argument("--device")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", help="fnmatch pattern on cell names")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)
    results = run_grid(args.grid, device=args.device, dry_run=args.dry_run, only=args.only, fail_fast=args.fail_fast)
    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"[matrix] {len(results)} cells: {counts}")
    return 1 if counts.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
