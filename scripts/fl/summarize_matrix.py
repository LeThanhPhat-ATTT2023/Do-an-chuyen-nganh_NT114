"""Summarize an FL matrix run into a CSV and markdown pivot tables.

    python scripts/fl/summarize_matrix.py --root outputs/fl/matrix_p0_defense
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from graphslm_ids.offline.federated.matrix import pivot_markdown, summarize, write_csv  # noqa: E402

DEFAULT_VALUES = [
    "final_test_macro_f1",
    "final_test_macro_f1_clean",
    "benign_false_rejection",
    "detect_tpr",
    "impact_final_test_macro_f1",
    "impact_final_test_macro_f1_clean",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize an FL experiment grid.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--out-csv")
    parser.add_argument("--out-md")
    parser.add_argument("--value", action="append", help="metric column to pivot (repeatable)")
    args = parser.parse_args(argv)
    root = Path(args.root)
    rows = summarize(root)
    write_csv(rows, Path(args.out_csv or root / "matrix_results.csv"))
    md = "\n".join(pivot_markdown(rows, value=v) for v in (args.value or DEFAULT_VALUES))
    Path(args.out_md or root / "matrix_results.md").write_text(md, encoding="utf-8")
    print(f"[summarize] {len(rows)} runs -> {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
