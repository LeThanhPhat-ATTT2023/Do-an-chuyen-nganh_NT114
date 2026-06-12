#!/usr/bin/env python3
"""Audit the EACS suspect/anchor masks against the clean answer key.

The EACS controller anchors every suspect-class flow that carries ANY
matching-family MITRE evidence (weight > 0) and treats the rest as suspects.
This script cross-tabulates both masks against the signature-isolated clean
answer key (the LNL oracle) to measure how trustworthy the anchor definition
actually is:

  * anchor precision  — of the flows EACS refuses to relabel, how many are
    REAL attacks?  Low precision = the model is force-fed wrong attack labels
    for the whole run (the candidate-set never touches anchors).
  * anchor recall     — of the real attacks, how many are protected as anchors?
    Low recall = true attacks sit in the suspect pool and get dragged to Benign.

Usage (on the box that has the artifact + answer key):
    python scripts/diagnostics/eacs_mask_audit.py \
        --npz outputs/v3_ob/graph.npz \
        --meta outputs/v3_ob/graph.meta.json \
        --clean-labels outputs/v3_ob/clean_eval_labels.npy \
        --mitre-dir data/mitre
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from graphslm_ids.offline.training.noise_consensus import (  # noqa: E402
    _FAMILY_ORDER,
    build_evidence_by_flow,
    class_to_family_from_csvs,
)

SUSPECT_CLASSES = ["CommandInjection", "XSS", "SqlInjection", "Uploading_Attack"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default="outputs/v3_ob/graph.npz")
    ap.add_argument("--meta", default="outputs/v3_ob/graph.meta.json")
    ap.add_argument("--clean-labels", default="outputs/v3_ob/clean_eval_labels.npy")
    ap.add_argument("--mitre-dir", default="data/mitre")
    ap.add_argument("--evidence-threshold", type=float, default=0.0,
                    help="anchor requires evidence weight > this (controller uses 0.0)")
    args = ap.parse_args()

    z = np.load(args.npz)
    meta = json.loads(Path(args.meta).read_text())
    label_mapping: dict[str, int] = meta["label_mapping"]
    num_classes = len(label_mapping)
    flow_y = torch.as_tensor(np.asarray(z["flow_y"], dtype=np.int64))
    clean = torch.as_tensor(np.asarray(np.load(args.clean_labels), dtype=np.int64))
    benign_id = label_mapping["Benign"]
    num_flows = int(flow_y.shape[0])

    # Rebuild evidence_by_flow exactly like evidence_table_from_artifact, from raw npz.
    contains = torch.as_tensor(np.asarray(z["contain_edge_index"], dtype=np.int64))
    family_to_col = {fam: i for i, fam in enumerate(_FAMILY_ORDER)}
    evidence_per_family: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for fam in _FAMILY_ORDER:
        k = f"evidence_{fam}_edge_index"
        if k not in z.files:
            continue
        edge = np.asarray(z[k], dtype=np.int64)
        attr_k = f"evidence_{fam}_edge_attr"
        if attr_k in z.files:
            attr = np.asarray(z[attr_k], dtype=np.float32).reshape(edge.shape[1], -1)
            w = torch.as_tensor(attr[:, 0])
        else:
            w = torch.ones(edge.shape[1], dtype=torch.float32)
        evidence_per_family[family_to_col[fam]] = (torch.as_tensor(edge[0]), w)
    ev = build_evidence_by_flow(contains, evidence_per_family, num_flows, len(_FAMILY_ORDER))

    class_to_family = class_to_family_from_csvs(args.mitre_dir, label_mapping, num_classes)
    sus_ids = sorted(label_mapping[c] for c in SUSPECT_CLASSES if c in label_mapping)
    in_cls = torch.isin(flow_y, torch.tensor(sus_ids, dtype=torch.long))
    fam = class_to_family[flow_y]
    has_ev = torch.zeros_like(in_cls)
    ok = fam >= 0
    idx_ok = torch.arange(num_flows)[ok]
    has_ev[idx_ok] = ev[idx_ok, fam[ok]] > args.evidence_threshold

    anchor = in_cls & has_ev
    suspect = in_cls & ~has_ev
    oracle_attack = in_cls & (clean == flow_y)       # answer key keeps the label
    oracle_noise = in_cls & (clean == benign_id)     # answer key demotes to Benign

    inv = {v: k for k, v in label_mapping.items()}
    print(f"threshold={args.evidence_threshold}  flows={num_flows}")
    print(f"{'class':<22}{'n':>7}{'true_atk':>9}{'anchors':>9}{'anchor∩atk':>11}"
          f"{'anch_prec':>10}{'anch_rec':>9}{'susp':>7}{'susp∩atk':>9}")
    tot = {k: 0 for k in ("n", "atk", "anc", "anc_atk", "sus", "sus_atk")}
    for cid in sus_ids:
        m = flow_y == cid
        atk = (oracle_attack & m)
        anc = (anchor & m)
        sus = (suspect & m)
        anc_atk = (anc & atk).sum().item()
        sus_atk = (sus & atk).sum().item()
        n, natk, nanc, nsus = m.sum().item(), atk.sum().item(), anc.sum().item(), sus.sum().item()
        prec = anc_atk / nanc if nanc else float("nan")
        rec = anc_atk / natk if natk else float("nan")
        print(f"{inv[cid]:<22}{n:>7}{natk:>9}{nanc:>9}{anc_atk:>11}"
              f"{prec:>10.3f}{rec:>9.3f}{nsus:>7}{sus_atk:>9}")
        tot["n"] += n; tot["atk"] += natk; tot["anc"] += nanc
        tot["anc_atk"] += anc_atk; tot["sus"] += nsus; tot["sus_atk"] += sus_atk
    prec = tot["anc_atk"] / tot["anc"] if tot["anc"] else float("nan")
    rec = tot["anc_atk"] / tot["atk"] if tot["atk"] else float("nan")
    print(f"{'TOTAL':<22}{tot['n']:>7}{tot['atk']:>9}{tot['anc']:>9}{tot['anc_atk']:>11}"
          f"{prec:>10.3f}{rec:>9.3f}{tot['sus']:>7}{tot['sus_atk']:>9}")
    print("\nReading: anch_prec is the fraction of anchored flows that are real attacks")
    print("(low -> the model is trained on wrong hard attack labels all run);")
    print("susp∩atk is real attacks left in the suspect pool (EACS may relabel them away).")


if __name__ == "__main__":
    main()
