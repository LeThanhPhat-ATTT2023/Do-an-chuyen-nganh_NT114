"""Build the eval-only CLEAN answer key for the LNL protocol.

Runs the existing signature-based attack-flow isolation
(`flow_attack_labeler.label_pcap_flows`) over the web-attack pcaps and maps the
resulting attack-flow keys onto a graph artifact's `flow_id_order`, producing a
clean label vector ALIGNED to `flow_y`. Flows of the 4 web classes whose
canonical 5-tuple carries a matching HTTP attack signature keep their label;
the rest of those classes' flows are demoted to Benign (they are background
IoT->cloud traffic — see docs/reports/2026-06-06-web-attack-encryption-ceiling.md
§3d). All other classes pass through unchanged.

The output is the GRADING KEY of the standard learning-with-noisy-labels
protocol (train on noisy, evaluate on clean). It is never used in training.

Usage (local, pcaps + meta required):
    python scripts/tools/extract_clean_eval_labels.py \
        --graph-meta outputs/v3_ob/graph.meta.json \
        --raw-root data/raw/14gb \
        --out-npy outputs/v3_ob/clean_eval_labels.npy \
        --out-audit outputs/v3_ob/clean_eval_labels.audit.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from graphslm_ids.offline.preprocessing.flow_attack_labeler import (
    BENIGN_LABEL,
    WEB_ATTACK_CLASSES,
    label_pcap_flows,
)


def label_of_flow_id(flow_id: str) -> str:
    """'Label|lo|hi|proto#seg.dir' -> 'Label'."""
    return flow_id.split("|", 1)[0]


def canonical_key_of_flow_id(flow_id: str) -> str:
    """'Label|lo|hi|proto#seg.dir' -> 'lo|hi|proto' (matches _canon_key)."""
    core = flow_id.split("|", 1)[1]
    return core.rsplit("#", 1)[0]


def clean_labels_from_attack_keys(
    flow_id_order: list[str],
    label_mapping: dict[str, int],
    attack_keys_by_class: dict[str, set[str]],
) -> tuple[np.ndarray, dict[str, int]]:
    """Clean label vector aligned to flow_id_order + per-class demotion audit.

    Only classes present in ``attack_keys_by_class`` are relabel-eligible; for
    those, a flow keeps its label iff its canonical key is in the class's attack
    key set, else it becomes Benign. Every other class passes through.
    """
    benign_id = label_mapping[BENIGN_LABEL]
    out = np.empty(len(flow_id_order), dtype=np.int64)
    audit: dict[str, int] = {}
    for i, fid in enumerate(flow_id_order):
        name = label_of_flow_id(fid)
        cls_id = label_mapping[name]
        keys = attack_keys_by_class.get(name)
        if keys is None:
            out[i] = cls_id
        elif canonical_key_of_flow_id(fid) in keys:
            out[i] = cls_id
        else:
            out[i] = benign_id
            audit[name] = audit.get(name, 0) + 1
    return out, audit


def collect_attack_keys(raw_root: Path) -> dict[str, set[str]]:
    """Run the signature isolation over every web-attack pcap under raw_root."""
    keys: dict[str, set[str]] = {}
    for cls in sorted(WEB_ATTACK_CLASSES):
        cls_dir = raw_root / cls
        pcaps = sorted(cls_dir.glob("*.pcap")) if cls_dir.exists() else []
        if not pcaps:
            print(f"[warn] no pcaps for {cls} under {cls_dir} — class skipped")
            continue
        cls_keys: set[str] = set()
        for pcap in pcaps:
            mapping, audit = label_pcap_flows(pcap, cls)
            cls_keys |= {k for k, v in mapping.items() if v == cls}
            print(f"[{cls}] {pcap.name}: {len(cls_keys)} attack keys, audit={audit}")
        keys[cls] = cls_keys
    return keys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph-meta", type=Path, required=True)
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--out-npy", type=Path, required=True)
    ap.add_argument("--out-audit", type=Path, required=True)
    args = ap.parse_args()

    meta = json.loads(args.graph_meta.read_text(encoding="utf-8"))
    flow_id_order = meta["flow_id_order"]
    label_mapping = meta["label_mapping"]

    attack_keys = collect_attack_keys(args.raw_root)
    clean, audit = clean_labels_from_attack_keys(
        flow_id_order, label_mapping, attack_keys
    )

    orig = np.array([label_mapping[label_of_flow_id(f)] for f in flow_id_order])
    n_changed = int((clean != orig).sum())
    args.out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_npy, clean)
    summary = {
        "n_flows": len(flow_id_order),
        "n_demoted_to_benign": n_changed,
        "demoted_per_class": audit,
        "attack_keys_per_class": {k: len(v) for k, v in attack_keys.items()},
        "graph_meta": str(args.graph_meta),
    }
    args.out_audit.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
