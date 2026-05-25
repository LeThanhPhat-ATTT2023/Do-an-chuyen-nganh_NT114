"""v2 PCAP extractor — keeps ALL packets, records TCP flags + ip_len + direction.

Replaces v1 `pcap_payload_extractor.py` which silently dropped:
  - TCP control packets (SYN/ACK/RST/FIN with no L4 payload) due to
    `include_empty_payload=False` default
  - All ICMP and non-TCP/UDP traffic
  - TCP flag bits (never recorded)

These were the chief root cause of the recon/scan cluster collapse in v1.
v2 keeps every IP packet that dpkt can parse and emits a tidy DataFrame ready
for flow assembly.
"""
from __future__ import annotations

import socket
from pathlib import Path

import dpkt
import pandas as pd

# pcap data-link-type values we know how to peel back to the IP layer.
_DLT_EN10MB = 1
_DLT_RAW = 12
_DLT_RAW2 = 101
_DLT_LINUX_SLL = 113

_PROTO_TCP = 0
_PROTO_UDP = 1
_PROTO_ICMP = 2
_PROTO_OTHER = 3

COLUMNS: list[str] = [
    "ts",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "proto",
    "flags",
    "ip_len",
    "payload_len",
    "label",
]


def _decode_ip(buf: bytes, dlt: int) -> object | None:
    """Peel the link-layer header and return the IP object, or None if unparseable."""
    try:
        if dlt == _DLT_EN10MB:
            return dpkt.ethernet.Ethernet(buf).data
        if dlt in (_DLT_RAW, _DLT_RAW2):
            return dpkt.ip.IP(buf)
        if dlt == _DLT_LINUX_SLL:
            # Linux cooked capture: 16-byte sll header then IP.
            return dpkt.ip.IP(buf[16:])
        # Unknown linktype — best-effort: try Ethernet first.
        return dpkt.ethernet.Ethernet(buf).data
    except Exception:
        return None


def extract_packets(
    pcap_path: Path, label: str, max_packets: int | None = None
) -> pd.DataFrame:
    """Parse a single pcap and return one row per IP packet.

    No filtering by payload length, no filtering by protocol — every parseable
    IP packet is emitted. Empty-payload TCP control packets and ICMP are kept;
    that's the point of v2.
    """
    rows: list[tuple] = []
    with open(pcap_path, "rb") as f:
        try:
            reader = dpkt.pcap.Reader(f)
            dlt = reader.datalink()
        except Exception:
            return pd.DataFrame(rows, columns=COLUMNS)
        for i, (ts, buf) in enumerate(reader):
            if max_packets is not None and i >= max_packets:
                break
            ip = _decode_ip(buf, dlt)
            if not isinstance(ip, dpkt.ip.IP):
                continue
            transport = ip.data
            sport = dport = -1
            flags = 0
            if isinstance(transport, dpkt.tcp.TCP):
                proto = _PROTO_TCP
                sport = int(transport.sport)
                dport = int(transport.dport)
                flags = int(transport.flags)
                plen = len(transport.data)
            elif isinstance(transport, dpkt.udp.UDP):
                proto = _PROTO_UDP
                sport = int(transport.sport)
                dport = int(transport.dport)
                plen = len(transport.data)
            elif isinstance(transport, dpkt.icmp.ICMP):
                proto = _PROTO_ICMP
                plen = len(bytes(transport.data))
            else:
                proto = _PROTO_OTHER
                plen = 0
            try:
                src = socket.inet_ntoa(ip.src)
                dst = socket.inet_ntoa(ip.dst)
            except Exception:
                continue
            rows.append(
                (
                    float(ts),
                    src,
                    dst,
                    sport,
                    dport,
                    proto,
                    flags,
                    int(ip.len),
                    int(plen),
                    label,
                )
            )
    return pd.DataFrame(rows, columns=COLUMNS)


def extract_packets_dir(
    raw_root: Path, max_per_class: int | None = None
) -> pd.DataFrame:
    """Iterate ``<raw_root>/<class>/*.pcap`` and concatenate results.

    Label is inferred from the immediate parent folder name (matches the
    project's per-pcap-folder labeling convention).
    """
    parts: list[pd.DataFrame] = []
    for cls_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        label = cls_dir.name
        for pcap in sorted(cls_dir.glob("*.pcap")):
            parts.append(
                extract_packets(pcap, label=label, max_packets=max_per_class)
            )
    if not parts:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(parts, ignore_index=True)
