#!/usr/bin/env python3
"""Build the EACS v2 anchor mask: HTTP-request packets with MITRE procedure hits.

The v1 anchor (any matching-family MSEE evidence weight > 0) is 84% noise: PMI
co-occurrence fires on campaign background, so 5.4k benign flows keep hard
attack labels all run and drown the ~1k real attacks 5:1 (see
scripts/diagnostics/eacs_mask_audit.py).

This tool anchors a web-attack flow iff one of its TCP payloads is an HTTP
REQUEST that matches a MITRE STIX procedure literal (the pipeline's own
Aho-Corasick matcher, MSEE source 2). Against the clean answer key this scores
precision 0.952 / recall 0.997 — but it never reads the answer key's
class-specific signatures: HTTP-method detection is generic protocol parsing
and the literals come from MITRE STIX. Train-time legitimate.

Output: a bool .npy aligned to graph.meta.json's flow_id_order + audit JSON.

    python scripts/tools/extract_eacs_anchor_mask.py \
        --graph-meta outputs/v3_ob/graph.meta.json \
        --raw-root data/raw \
        --stix data/mitre/enterprise-attack.json \
        --out-npy outputs/v3_ob/eacs_anchor_mask.npy \
        --out-audit outputs/v3_ob/eacs_anchor_mask.audit.json
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

import dpkt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from graphslm_ids.offline.preprocessing.flow_attack_labeler import (  # noqa: E402
    WEB_ATTACK_CLASSES,
    _canon_key,
    _HTTP_METHOD,
)
from graphslm_ids.offline.preprocessing.procedure_matcher import (  # noqa: E402
    ProcedureMatcher,
)


def label_of_flow_id(flow_id: str) -> str:
    """'Label|lo|hi|proto#seg.dir' -> 'Label'."""
    return flow_id.split("|", 1)[0]


def canonical_key_of_flow_id(flow_id: str) -> str:
    """'Label|lo|hi|proto#seg.dir' -> 'lo|hi|proto' (matches _canon_key)."""
    core = flow_id.split("|", 1)[1]
    return core.rsplit("#", 1)[0]


def collect_anchor_keys(
    raw_root: Path, matcher: ProcedureMatcher
) -> dict[str, set[str]]:
    """Per web class: canonical flow keys with an HTTP-request procedure hit."""
    out: dict[str, set[str]] = {}
    for cls in sorted(WEB_ATTACK_CLASSES):
        keys: set[str] = set()
        for pcap in sorted((raw_root / cls).glob("*.pcap")):
            with open(pcap, "rb") as fh:
                try:
                    reader = dpkt.pcap.Reader(fh)
                    dlt = reader.datalink()
                except ValueError:
                    fh.seek(0)
                    reader = dpkt.pcapng.Reader(fh)
                    dlt = None
                for _ts, buf in reader:
                    try:
                        ip = (
                            dpkt.ip.IP(buf)
                            if (dlt is not None and dlt != dpkt.pcap.DLT_EN10MB)
                            else dpkt.ethernet.Ethernet(buf).data
                        )
                    except Exception:
                        continue
                    if not isinstance(ip, dpkt.ip.IP) or not isinstance(
                        ip.data, dpkt.tcp.TCP
                    ):
                        continue
                    l4 = ip.data
                    data = bytes(l4.data)
                    if not data or not _HTTP_METHOD.match(data):
                        continue
                    try:
                        key = _canon_key(
                            socket.inet_ntoa(ip.src), l4.sport,
                            socket.inet_ntoa(ip.dst), l4.dport, 0,
                        )
                    except Exception:
                        continue
                    if key in keys:
                        continue
                    if matcher.match(data.lower()):
                        keys.add(key)
        out[cls] = keys
        print(f"[{cls}] anchor keys: {len(keys)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph-meta", type=Path, required=True)
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--stix", type=Path, default=Path("data/mitre/enterprise-attack.json"))
    ap.add_argument("--out-npy", type=Path, required=True)
    ap.add_argument("--out-audit", type=Path, required=True)
    args = ap.parse_args()

    meta = json.loads(args.graph_meta.read_text(encoding="utf-8"))
    flow_id_order: list[str] = meta["flow_id_order"]

    matcher = ProcedureMatcher(args.stix)
    anchor_keys = collect_anchor_keys(args.raw_root, matcher)

    mask = np.zeros(len(flow_id_order), dtype=bool)
    per_class: dict[str, int] = {}
    for i, fid in enumerate(flow_id_order):
        name = label_of_flow_id(fid)
        keys = anchor_keys.get(name)
        if keys and canonical_key_of_flow_id(fid) in keys:
            mask[i] = True
            per_class[name] = per_class.get(name, 0) + 1

    args.out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_npy, mask)
    summary = {
        "n_flows": len(flow_id_order),
        "n_anchored": int(mask.sum()),
        "anchored_per_class": per_class,
        "anchor_keys_per_class": {k: len(v) for k, v in anchor_keys.items()},
        "graph_meta": str(args.graph_meta),
    }
    args.out_audit.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
