"""Attack-flow isolation for web-attack captures.

Root cause (see docs/reports/2026-06-06-web-attack-encryption-ceiling.md):
web-attack pcaps (CommandInjection/XSS/SqlInjection/Uploading_Attack) are ~95%
background IoT->cloud traffic, all mislabeled as the attack by per-pcap labeling.
The TRUE attack is plaintext HTTP to a local web app (DVWA). Isolating those flows
lifts web-cluster separability from 0.51 to 0.95.

This module re-labels flows in web-attack captures: a flow keeps the attack label
ONLY if it carries a plaintext HTTP request matching that class's attack signature;
otherwise it becomes Benign. Determination is signature-based (URL path / payload
pattern), NOT a hardcoded victim IP -- so it generalizes beyond this testbed.

Non-web captures (DDoS/Recon/Mirai/Benign) are NOT touched: their whole capture IS
the attack, so per-pcap labeling is correct there.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import dpkt

BENIGN_LABEL = "Benign"

# Classes whose capture is polluted with background and whose attack is an HTTP
# signature against a web app. Only these get the isolation treatment.
WEB_ATTACK_CLASSES = frozenset(
    {"CommandInjection", "XSS", "SqlInjection", "Uploading_Attack"}
)

_HTTP_METHOD = re.compile(rb"^(GET|POST|PUT|DELETE|HEAD|OPTIONS|PATCH) ")

# Per-class signatures, matched on the lowercased HTTP request (request-line +
# headers + body). Two tiers: (a) DVWA endpoint path (testbed-specific, high
# precision) and (b) payload attack pattern (generalizes to any web app).
_SIGNATURES: dict[str, list[bytes]] = {
    "CommandInjection": [
        rb"/exec", rb";cat", rb";ls", rb"|sh", rb"|bash", rb"$(", rb"&&",
        rb"`", rb"/bin/", rb"/etc/passwd", rb"whoami", rb"ping -c", rb"uname",
    ],
    "XSS": [
        rb"/xss", rb"<script", rb"%3cscript", rb"onerror=", rb"onload=",
        rb"javascript:", rb"alert(", rb"<svg", rb"<img", rb"<iframe",
        rb"document.cookie",
    ],
    "SqlInjection": [
        rb"/sqli", rb"union select", rb"union%20select", rb"or 1=1",
        rb"or%201=1", rb"' or '", rb"%27+or", rb"%27or", rb"information_schema",
        rb"sleep(", rb"concat(", rb"--", rb"/*",
    ],
    "Uploading_Attack": [
        rb"/upload", rb"multipart/form-data", rb"content-disposition",
        rb"filename=", rb"boundary=", rb".php", rb".jsp", rb".asp",
    ],
}


def _canon_key(src_ip: str, sport: int, dst_ip: str, dport: int, proto: int) -> str:
    """Canonical bidirectional 5-tuple key WITHOUT the label/segment, matching
    the (lo|hi|proto) core used by flows.assign_flows."""
    a = f"{src_ip}:{sport}"
    b = f"{dst_ip}:{dport}"
    lo, hi = (a, b) if a <= b else (b, a)
    return f"{lo}|{hi}|{proto}"


def request_matches(payload_lower: bytes, attack_class: str) -> bool:
    """True if the HTTP request payload matches the attack class signature."""
    sigs = _SIGNATURES.get(attack_class)
    if not sigs:
        return False
    return any(s in payload_lower for s in sigs)


def label_pcap_flows(
    pcap_path: Path, original_label: str, max_packets: int | None = None
) -> tuple[dict[str, str], dict[str, int]]:
    """Return (flow_key -> new_label, audit_counts) for one pcap.

    For non-web-attack captures this is a no-op: every flow keeps ``original_label``
    (caller may skip calling it). For web-attack captures, a flow keeps the attack
    label iff any of its packets is a plaintext HTTP request matching the signature;
    all other flows become Benign.
    """
    if original_label not in WEB_ATTACK_CLASSES:
        return {}, {"skipped_non_web": 1}

    # First pass: find flows containing a matching attack request.
    attack_flows: set[str] = set()
    all_flows: set[str] = set()
    with open(pcap_path, "rb") as fh:
        try:
            reader = dpkt.pcap.Reader(fh)
            dlt = reader.datalink()
        except ValueError:
            fh.seek(0)
            reader = dpkt.pcapng.Reader(fh)
            dlt = None
        for i, (_ts, buf) in enumerate(reader):
            if max_packets is not None and i >= max_packets:
                break
            try:
                if dlt is not None and dlt != dpkt.pcap.DLT_EN10MB:
                    ip = dpkt.ip.IP(buf)
                else:
                    ip = dpkt.ethernet.Ethernet(buf).data
            except Exception:
                continue
            if not isinstance(ip, dpkt.ip.IP):
                continue
            l4 = ip.data
            if not isinstance(l4, dpkt.tcp.TCP):
                continue
            import socket
            try:
                src_ip = socket.inet_ntoa(ip.src)
                dst_ip = socket.inet_ntoa(ip.dst)
            except Exception:
                continue
            key = _canon_key(src_ip, l4.sport, dst_ip, l4.dport, 0)  # proto 0=TCP
            all_flows.add(key)
            data = bytes(l4.data)
            if data and _HTTP_METHOD.match(data):
                if request_matches(data.lower(), original_label):
                    attack_flows.add(key)

    mapping = {k: (original_label if k in attack_flows else BENIGN_LABEL)
               for k in all_flows}
    audit = {
        "original_label": original_label,
        "total_tcp_flows": len(all_flows),
        "true_attack_flows": len(attack_flows),
        "relabeled_benign": len(all_flows) - len(attack_flows),
    }
    return mapping, audit


def relabel_packets_df(packets_df, log=None):
    """Relabel a packets DataFrame in place: in web-attack captures, a flow
    (canonical 5-tuple) keeps its attack label only if ANY of its packets is a
    plaintext HTTP request matching the class signature; otherwise -> Benign.

    Expects columns: src_ip, dst_ip, src_port, dst_port, proto, label, payload
    (payload = bytes, may be truncated to ~256B; HTTP request-line + URL path
    signatures live in the first bytes so truncation is fine). Returns the same
    DataFrame plus an audit dict {class: {total, attack, benign}}.
    """
    import numpy as np
    import pandas as pd

    if "label" not in packets_df.columns or "payload" not in packets_df.columns:
        return packets_df, {"error": "missing label/payload column"}

    df = packets_df
    labels = df["label"].astype(str)
    web_class_mask = labels.isin(WEB_ATTACK_CLASSES)
    if not web_class_mask.any():
        return df, {"note": "no web-attack-class packets"}
    # Attack requests only live in plaintext TCP/HTTP packets.
    atk_scan_mask = web_class_mask & (df["proto"].astype(int) == 0)

    # Canonical bidirectional key (matches flows.assign_flows lo|hi ordering).
    a = df["src_ip"].astype(str) + ":" + df["src_port"].astype(str)
    b = df["dst_ip"].astype(str) + ":" + df["dst_port"].astype(str)
    canon = np.minimum(a, b) + "|" + np.maximum(a, b)
    # group key includes label so different captures never merge.
    gkey = labels + "||" + canon

    sub = df.loc[atk_scan_mask]
    sub_labels = labels.loc[atk_scan_mask]
    sub_gkey = gkey.loc[atk_scan_mask]

    def _is_attack(row_payload, cls) -> bool:
        if not isinstance(row_payload, (bytes, bytearray)) or not row_payload:
            return False
        if not _HTTP_METHOD.match(row_payload):
            return False
        return request_matches(bytes(row_payload).lower(), cls)

    is_atk = [
        _is_attack(p, c)
        for p, c in zip(sub["payload"].tolist(), sub_labels.tolist())
    ]
    is_atk = pd.Series(is_atk, index=sub.index)
    # A connection is an attack if ANY of its packets matched.
    attack_keys = set(sub_gkey[is_atk].tolist())

    # Any web-class packet (incl. UDP/non-HTTP background) whose connection is
    # NOT an attack -> Benign.
    demote_mask = web_class_mask & ~gkey.isin(attack_keys)
    audit = {}
    for cls in sorted(WEB_ATTACK_CLASSES):
        cls_mask = labels == cls
        tot = int(cls_mask.sum())
        if tot == 0:
            continue
        dem = int((cls_mask & demote_mask).sum())
        audit[cls] = {"total_pkts": tot, "kept_attack_pkts": tot - dem,
                      "relabeled_benign_pkts": dem}
    df.loc[demote_mask, "label"] = BENIGN_LABEL
    if log is not None:
        log.info("[attack-isolation] relabeled background->Benign: %s", audit)
    return df, audit


def _validate() -> None:
    """Standalone validation on the raw web-attack pcaps."""
    import sys
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset/raw")
    files = {
        "CommandInjection": root / "CommandInjection" / "CommandInjection.pcap",
        "XSS": root / "XSS" / "XSS.pcap",
        "SqlInjection": root / "SqlInjection" / "SqlInjection.pcap",
        "Uploading_Attack": root / "Uploading_Attack" / "Uploading_Attack.pcap",
    }
    for cls, p in files.items():
        if not p.exists():
            print(f"[skip] {cls}: {p} missing")
            continue
        mapping, audit = label_pcap_flows(p, cls)
        dist = Counter(mapping.values())
        print(f"[{cls:18s}] {audit} | new-label dist: {dict(dist)}")


if __name__ == "__main__":
    _validate()
