#!/usr/bin/env python3
"""Can MITRE procedure-literal matches anchor true web attacks?

The MSEE ensemble collapses its three sources into one scalar edge weight, so
the artifact cannot tell a high-precision procedure hit from a diffuse PMI
co-occurrence — and the EACS anchor built on that scalar is 84% noise.

This test measures, per web-attack pcap, how well a PROCEDURE-ONLY anchor
would do: a flow is anchored iff any of its TCP payloads matches a MITRE STIX
procedure literal (Aho-Corasick, same matcher the pipeline uses). Oracle =
the signature-isolated clean answer key (label_pcap_flows).

Run locally:  python scripts/diagnostics/eacs_procedure_anchor_test.py
"""
from __future__ import annotations

import socket
import sys
from collections import defaultdict
from pathlib import Path

import dpkt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from graphslm_ids.offline.preprocessing.flow_attack_labeler import (  # noqa: E402
    WEB_ATTACK_CLASSES,
    _canon_key,
    label_pcap_flows,
)
from graphslm_ids.offline.preprocessing.procedure_matcher import (  # noqa: E402
    ProcedureMatcher,
)

RAW_ROOT = Path("data/raw")
STIX = Path("data/mitre/enterprise-attack.json")


def flow_payload_hits(pcap_path: Path, matcher: ProcedureMatcher) -> dict[str, int]:
    """flow canonical key -> number of packets with >=1 procedure hit."""
    hits: dict[str, int] = defaultdict(int)
    with open(pcap_path, "rb") as fh:
        try:
            reader = dpkt.pcap.Reader(fh)
            dlt = reader.datalink()
        except ValueError:
            fh.seek(0)
            reader = dpkt.pcapng.Reader(fh)
            dlt = None
        for _ts, buf in reader:
            try:
                if dlt is not None and dlt != dpkt.pcap.DLT_EN10MB:
                    ip = dpkt.ip.IP(buf)
                else:
                    ip = dpkt.ethernet.Ethernet(buf).data
            except Exception:
                continue
            if not isinstance(ip, dpkt.ip.IP) or not isinstance(ip.data, dpkt.tcp.TCP):
                continue
            l4 = ip.data
            data = bytes(l4.data)
            if not data:
                continue
            try:
                key = _canon_key(
                    socket.inet_ntoa(ip.src), l4.sport,
                    socket.inet_ntoa(ip.dst), l4.dport, 0,
                )
            except Exception:
                continue
            if matcher.match(data.lower()):
                hits[key] += 1
    return dict(hits)


def main() -> None:
    matcher = ProcedureMatcher(STIX)
    tot = {"anc": 0, "anc_atk": 0, "atk": 0, "flows": 0}
    print(f"{'class':<20}{'flows':>7}{'attacks':>8}{'proc-anchored':>14}{'x-attack':>9}"
          f"{'precision':>10}{'recall':>8}")
    for cls in sorted(WEB_ATTACK_CLASSES):
        pcaps = sorted((RAW_ROOT / cls).glob("*.pcap"))
        for pcap in pcaps:
            mapping, _ = label_pcap_flows(pcap, cls)
            attack_keys = {k for k, v in mapping.items() if v == cls}
            hits = flow_payload_hits(pcap, matcher)
            anchored = set(hits)
            inter = anchored & attack_keys
            n_f, n_a, n_anc, n_i = len(mapping), len(attack_keys), len(anchored), len(inter)
            prec = n_i / n_anc if n_anc else float("nan")
            rec = n_i / n_a if n_a else float("nan")
            print(f"{cls:<20}{n_f:>7}{n_a:>8}{n_anc:>14}{n_i:>9}{prec:>10.3f}{rec:>8.3f}")
            tot["flows"] += n_f; tot["atk"] += n_a
            tot["anc"] += n_anc; tot["anc_atk"] += n_i
    prec = tot["anc_atk"] / tot["anc"] if tot["anc"] else float("nan")
    rec = tot["anc_atk"] / tot["atk"] if tot["atk"] else float("nan")
    print(f"{'TOTAL':<20}{tot['flows']:>7}{tot['atk']:>8}{tot['anc']:>14}"
          f"{tot['anc_atk']:>9}{prec:>10.3f}{rec:>8.3f}")


if __name__ == "__main__":
    main()
