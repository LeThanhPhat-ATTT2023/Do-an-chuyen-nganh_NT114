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

import multiprocessing as _mp
import os
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


def _extract_pcap_worker(args: tuple) -> pd.DataFrame:
    """Top-level worker (picklable on Windows spawn) — parse one PCAP file."""
    pcap_path_str, label, max_packets = args
    return extract_packets(Path(pcap_path_str), label=label, max_packets=max_packets)


def _payload_class_worker(args: tuple) -> str:
    """Stage-2 worker: re-read one class dir's PCAPs and write payload bytes
    directly into a pre-created memmap (shape ``(n_rows, payload_length)``).

    ``row_index_slice`` maps local packet position (int) → list of row indices
    in the output matrix that correspond to that packet.  Workers write to
    non-overlapping row ranges so no locking is needed.

    Returns ``label`` for progress logging.
    """
    import numpy as np

    cls_dir_str, label, row_index_slice, payload_length, mmap_path, n_rows = args
    out = np.memmap(mmap_path, dtype=np.uint8, mode="r+", shape=(n_rows, payload_length))

    running_pos = 0
    for pcap in sorted(Path(cls_dir_str).glob("*.pcap")):
        try:
            fh = open(pcap, "rb")
        except OSError:
            continue
        with fh:
            try:
                reader = dpkt.pcap.Reader(fh)
                dlt = reader.datalink()
            except Exception:
                continue
            for _ts, buf in reader:
                ip = _decode_ip(buf, dlt)
                if not isinstance(ip, dpkt.ip.IP):
                    continue
                rows = row_index_slice.get(running_pos)
                if rows:
                    t = ip.data
                    if isinstance(t, dpkt.tcp.TCP):
                        payload = bytes(t.data)
                    elif isinstance(t, dpkt.udp.UDP):
                        payload = bytes(t.data)
                    elif isinstance(t, dpkt.icmp.ICMP):
                        payload = bytes(t.data)
                    else:
                        payload = b""
                    if payload:
                        vec = np.zeros(payload_length, dtype=np.uint8)
                        k = min(payload_length, len(payload))
                        vec[:k] = np.frombuffer(payload[:k], dtype=np.uint8)
                        for r in rows:
                            out[r] = vec
                running_pos += 1
    out.flush()
    return label


def extract_packets_dir(
    raw_root: Path,
    max_per_class: int | None = None,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """Iterate ``<raw_root>/<class>/*.pcap`` and concatenate results — parallel.

    Each PCAP is parsed in a separate worker process (one per file). With 18
    PCAPs and 16 CPUs the wall-clock time drops from ~N×single to ~ceil(18/16)
    rounds, limited only by disk throughput. Label is inferred from the
    immediate parent folder name.
    """
    tasks: list[tuple] = []
    for cls_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        label = cls_dir.name
        for pcap in sorted(cls_dir.glob("*.pcap")):
            tasks.append((str(pcap), label, max_per_class))

    if not tasks:
        return pd.DataFrame(columns=COLUMNS)

    if n_workers is None:
        # Auto-scale: read available RAM at runtime so the worker count adapts
        # to the machine.  On a 16 GB laptop this stays at ~1-2; on a 64 GB
        # cloud instance (g6e.2xlarge) it scales up to use all 8 CPUs.
        from graphslm_ids.offline.preprocessing._resource import auto_pcap_workers
        n_workers = auto_pcap_workers(len(tasks))

    ctx = _mp.get_context("spawn")
    parts: list[pd.DataFrame] = []
    # Use imap so finished worker RAM is freed before the next task starts.
    with ctx.Pool(processes=n_workers) as pool:
        for df in pool.imap(_extract_pcap_worker, tasks):
            if not df.empty:
                parts.append(df)

    if not parts:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(parts, ignore_index=True)
