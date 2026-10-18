"""Export the FL client x class flow-count table (CSV) and its heatmap (PNG) for the thesis.

    python scripts/fl/export_partition_table.py --config configs/fl/fl_base.yaml \
        --out-csv outputs/fl/partition_table.csv --out-png outputs/fl/partition_table.png \
        --set partition.dominance=0.5

Builds the partition exactly as ``run_experiment`` does (same train split, same
``make_partition`` call) but trains nothing. Rows are clients (1-based), columns the classes in
class-id order; the ``group`` column is the gateway group (explicit scheme only).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from graphslm_ids.offline.federated.config import apply_overrides, load_fl_config, parse_set_args  # noqa: E402
from graphslm_ids.offline.federated.context import load_context  # noqa: E402
from graphslm_ids.offline.federated.partition import group_members, make_partition  # noqa: E402

# Chart ink and a single-hue (blue) sequential ramp, light surface: magnitude reads light -> dark.
_SURFACE, _INK, _INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the client x class partition table (CSV + heatmap).")
    parser.add_argument("--config", required=True, help="FL YAML config (configs/fl/*.yaml)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="dotted override, repeatable")
    parser.add_argument("--out-csv", default="outputs/fl/partition_table.csv")
    parser.add_argument("--out-png", default="outputs/fl/partition_table.png")
    return parser


def _group_of_client(cfg: dict[str, Any], scheme: str, num_clients: int) -> list[str]:
    """Gateway group name per client; empty strings unless the scheme is ``explicit``."""
    if scheme != "explicit":
        return [""] * num_clients
    owner = [""] * num_clients
    for name, clients in group_members(cfg["partition"]["groups"], num_clients).items():
        for k in clients:
            owner[k] = name
    return owner


def write_csv(path: Path, class_names: list[str], counts: np.ndarray, groups: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["client", "group", *class_names, "total"])
        for k, row in enumerate(counts):
            writer.writerow([k + 1, groups[k], *[int(v) for v in row], int(row.sum())])


def write_heatmap(path: Path, class_names: list[str], counts: np.ndarray, groups: list[str], title: str) -> bool:
    """Annotated client x class heatmap; returns False (after a note) when matplotlib is missing."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping the heatmap (the CSV is written).")
        return False

    k_total, c_total = counts.shape
    fig, ax = plt.subplots(figsize=(0.62 * c_total + 3.2, 0.42 * k_total + 2.4), facecolor=_SURFACE)
    ax.set_facecolor(_SURFACE)
    image = ax.imshow(counts, cmap="Blues", aspect="auto", vmin=0)
    top = float(counts.max()) if counts.size else 1.0
    for k in range(k_total):
        for c in range(c_total):
            value = int(counts[k, c])
            dark_cell = top > 0 and value / top > 0.55
            ax.text(c, k, str(value), ha="center", va="center", fontsize=7, color="white" if dark_cell else _INK)
    ax.set_xticks(range(c_total), labels=class_names, rotation=60, ha="right", fontsize=8, color=_INK)
    ylabels = [f"C{k + 1} ({groups[k]})" if groups[k] else f"C{k + 1}" for k in range(k_total)]
    ax.set_yticks(range(k_total), labels=ylabels, fontsize=8, color=_INK)
    for k in range(1, k_total):  # a rule between gateway groups
        if groups[k] != groups[k - 1]:
            ax.axhline(k - 0.5, color=_INK_MUTED, linewidth=1.2)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlabel("class", color=_INK_MUTED)
    ax.set_ylabel("client (gateway group)", color=_INK_MUTED)
    ax.set_title(title, color=_INK, fontsize=11, loc="left")
    cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("train flows", color=_INK_MUTED)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=_INK_MUTED, length=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=_SURFACE)
    plt.close(fig)
    return True


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = apply_overrides(load_fl_config(args.config), parse_set_args(args.set))
    ctx = load_context(cfg)
    part = make_partition(
        ctx.train_ids, ctx.labels, cfg["partition"], label_names=ctx.label_names, order=ctx.flow_order
    )
    counts = part.class_histogram(ctx.labels, ctx.num_classes)
    class_names = [ctx.label_names.get(c, str(c)) for c in range(ctx.num_classes)]
    groups = _group_of_client(cfg, part.scheme, part.num_clients)
    out_csv = Path(args.out_csv)
    write_csv(out_csv, class_names, counts, groups)
    print(f"wrote {out_csv} ({part.num_clients} clients x {ctx.num_classes} classes)")
    title = f"Training flows per client and class ({part.scheme}"
    title += f", dominance {part.params['dominance']:g})" if "dominance" in part.params else ")"
    if write_heatmap(Path(args.out_png), class_names, counts, groups, title):
        print(f"wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
