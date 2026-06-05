#!/usr/bin/env python
"""Evidence-edge coverage per flow class.

For each flow class, compute the fraction of that class's packets that carry
each of the five typed ``packet -> evidence_{family} -> technique`` edge types:

    evidence_injection / evidence_command_exec / evidence_file_upload /
    evidence_recon / evidence_c2_beacon

This is a VALIDATION tool. After rebuilding the v3 graph artifact, the
web-attack classes should become discriminative via their *dominant* evidence
family. The key acceptance signal is the "web cluster check":

    CommandInjection -> evidence_command_exec
    Uploading_Attack -> evidence_file_upload
    XSS              -> evidence_injection

If those three classes share the same dominant family, the rebuild did NOT make
the web-attack cluster separable and something is wrong upstream.

The script is deterministic (no randomness) and reads the artifact through the
same loader the trainer uses: ``load_v3_artifact(..., add_reverse_edges=False)``.

Example
-------
    D:\\v\\nt114\\Scripts\\python.exe scripts/diagnostics/evidence_coverage.py ^
        --graph outputs/v3/graph.npz ^
        --graph-meta outputs/v3/graph.meta.json ^
        --out outputs/v3/evidence_coverage.json

    :: Restrict to the web-attack cluster
    D:\\v\\nt114\\Scripts\\python.exe scripts/diagnostics/evidence_coverage.py ^
        --graph outputs/v3/graph.npz ^
        --graph-meta outputs/v3/graph.meta.json ^
        --out outputs/v3/evidence_coverage_web.json ^
        --classes CommandInjection,XSS,Uploading_Attack
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from graphslm_ids.offline.training.hetero_graph_artifact import load_v3_artifact

# The five typed evidence edge types, in a fixed (deterministic) display order.
EV_TYPES: tuple[str, ...] = (
    "evidence_injection",
    "evidence_command_exec",
    "evidence_file_upload",
    "evidence_recon",
    "evidence_c2_beacon",
)

# Web-attack cluster classes and their EXPECTED dominant evidence family.
# Used only to render the acceptance "web cluster check" line.
WEB_CLUSTER: dict[str, str] = {
    "CommandInjection": "evidence_command_exec",
    "Uploading_Attack": "evidence_file_upload",
    "XSS": "evidence_injection",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="evidence_coverage.py",
        description=(
            "Compute per-class packet coverage for each typed evidence-edge "
            "family in a v3 graph artifact."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--graph",
        type=Path,
        required=True,
        help="Path to the v3 graph artifact (graph.npz).",
    )
    parser.add_argument(
        "--graph-meta",
        type=Path,
        required=True,
        help="Path to the graph metadata sidecar (graph.meta.json).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Path to write the JSON coverage report.",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help=(
            "Optional comma-separated class names to restrict the report to. "
            "Defaults to ALL classes in label_mapping, sorted by name."
        ),
    )
    return parser.parse_args(argv)


def compute_coverage(
    graph: Path,
    graph_meta: Path,
    requested_classes: list[str] | None,
) -> dict:
    """Return ``{class_name: {n_pkt, coverage: {et: pct}, dominant: et}}``.

    Classes with zero packets are recorded with ``n_pkt == 0`` and a ``None``
    dominant family so the caller can skip them with a note.
    """
    art = load_v3_artifact(
        graph_npz=graph,
        graph_meta_json=graph_meta,
        add_reverse_edges=False,
    )

    ei = art.edge_index
    flow_y = np.asarray(art.flow_y, dtype=np.int64)
    label_mapping: dict[str, int] = dict(art.metadata["label_mapping"])
    num_packets = int(art.metadata["num_packets"])

    # Map each packet -> its owning flow via flow--contains-->packet edges.
    fc = ei[("flow", "contains", "packet")]
    flow_of_pkt = np.full(num_packets, -1, dtype=np.int64)
    flow_of_pkt[fc[1]] = fc[0]

    # Boolean coverage mask per evidence type: True where a packet is the
    # source of at least one edge of that type.
    pkt_has: dict[str, np.ndarray] = {}
    for et in EV_TYPES:
        mask = np.zeros(num_packets, dtype=bool)
        edge = ei.get(("packet", et, "technique"))
        if edge is not None and edge.shape[1] > 0:
            mask[np.unique(edge[0])] = True
        pkt_has[et] = mask

    # Class label per packet (-1 for packets not attached to a labelled flow).
    cls_of_pkt = np.where(
        flow_of_pkt >= 0,
        flow_y[np.clip(flow_of_pkt, 0, None)],
        -1,
    )

    # Resolve which classes to report on.
    if requested_classes is None:
        class_names = sorted(label_mapping.keys())
    else:
        class_names = requested_classes

    report: dict[str, dict] = {}
    for name in class_names:
        if name not in label_mapping:
            report[name] = {
                "n_pkt": 0,
                "coverage": {et: 0.0 for et in EV_TYPES},
                "dominant": None,
                "note": "class name not found in label_mapping",
            }
            continue

        idx = label_mapping[name]
        class_mask = cls_of_pkt == idx
        n_pkt = int(class_mask.sum())

        if n_pkt == 0:
            report[name] = {
                "n_pkt": 0,
                "coverage": {et: 0.0 for et in EV_TYPES},
                "dominant": None,
                "note": "no packets for this class",
            }
            continue

        coverage = {
            et: 100.0 * float((pkt_has[et] & class_mask).sum()) / n_pkt
            for et in EV_TYPES
        }
        # argmax over the fixed EV_TYPES order; ties resolve to the first
        # (earliest) family deterministically.
        dominant = max(EV_TYPES, key=lambda et: coverage[et])
        report[name] = {
            "n_pkt": n_pkt,
            "coverage": coverage,
            "dominant": dominant,
        }

    return report


def _short(et: str) -> str:
    """Strip the ``evidence_`` prefix for compact column headers."""
    return et[len("evidence_"):] if et.startswith("evidence_") else et


def print_table(report: dict) -> None:
    """Print a readable, aligned coverage table."""
    name_w = max([len("class")] + [len(n) for n in report]) if report else len("class")
    col_w = 12

    header = "class".ljust(name_w)
    for et in EV_TYPES:
        header += "  " + _short(et).rjust(col_w)
    header += "  " + "n_pkt".rjust(12)
    print(header)
    print("-" * len(header))

    for name in report:
        row_data = report[name]
        line = name.ljust(name_w)
        if row_data["n_pkt"] == 0:
            for _ in EV_TYPES:
                line += "  " + "-".rjust(col_w)
            line += "  " + "0".rjust(12)
            note = row_data.get("note", "skipped")
            line += f"   ({note})"
            print(line)
            continue
        for et in EV_TYPES:
            pct = row_data["coverage"][et]
            line += "  " + f"{pct:.1f}%".rjust(col_w)
        line += "  " + str(row_data["n_pkt"]).rjust(12)
        print(line)


def print_web_cluster_check(report: dict) -> None:
    """Print the acceptance signal for the web-attack cluster classes."""
    print()
    print("=== web cluster check ===")
    seen_dominant: list[str] = []
    for name, expected in WEB_CLUSTER.items():
        if name not in report:
            print(f"  {name:<18} : (not in report)")
            continue
        row_data = report[name]
        if row_data["n_pkt"] == 0:
            note = row_data.get("note", "no packets")
            print(f"  {name:<18} : (skipped — {note})")
            continue
        dominant = row_data["dominant"]
        pct = row_data["coverage"][dominant]
        status = "OK" if dominant == expected else "MISMATCH"
        print(
            f"  {name:<18} : dominant={_short(dominant):<13} "
            f"({pct:.1f}%)  expected={_short(expected):<13} [{status}]"
        )
        seen_dominant.append(dominant)

    distinct = len(set(seen_dominant))
    if seen_dominant:
        if distinct == len(seen_dominant):
            print(
                f"  -> {distinct} distinct dominant families across "
                f"{len(seen_dominant)} web classes: cluster is discriminative."
            )
        else:
            print(
                f"  -> only {distinct} distinct dominant families across "
                f"{len(seen_dominant)} web classes: cluster NOT discriminative."
            )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    requested_classes: list[str] | None = None
    if args.classes is not None:
        requested_classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    report = compute_coverage(
        graph=args.graph,
        graph_meta=args.graph_meta,
        requested_classes=requested_classes,
    )

    print_table(report)
    print_web_cluster_check(report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print()
    print(f"Wrote coverage report to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
