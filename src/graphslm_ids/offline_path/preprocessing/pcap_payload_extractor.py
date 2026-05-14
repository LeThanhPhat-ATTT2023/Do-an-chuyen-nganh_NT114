from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
from scapy.layers.inet import IP, TCP, UDP
from scapy.packet import Raw
from scapy.utils import PcapReader

import socket as _socket

try:
    import dpkt.pcap as _dpkt_pcap
    import dpkt.ethernet as _dpkt_eth
    import dpkt.ip as _dpkt_ip
    import dpkt.tcp as _dpkt_tcp
    import dpkt.udp as _dpkt_udp
    _DPKT_AVAILABLE = True
    try:
        import dpkt.pcapng as _dpkt_pcapng
        _DPKT_HAS_PCAPNG = True
    except ImportError:
        _DPKT_HAS_PCAPNG = False
except ImportError:
    _DPKT_AVAILABLE = False
    _DPKT_HAS_PCAPNG = False

# PCAP datalink type constants
_DLT_EN10MB = 1    # Standard Ethernet
_DLT_RAW = 101     # Raw IPv4 (Linux/BSD)
_DLT_RAW2 = 228    # LINKTYPE_IPV4 (macOS)


@dataclass
class PacketRecord:
    pcap_file: str
    packet_index: int
    timestamp: float
    label: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    payload_len_raw: int
    payload_256: np.ndarray


METADATA_COLUMNS: tuple[str, ...] = (
    "pcap_file",
    "packet_index",
    "timestamp",
    "label",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "payload_len_raw",
)


@dataclass
class PayloadDatasetDiskResult:
    payload_path: Path
    metadata_path: Path
    num_packets: int
    label_counts: dict[str, int]
    protocol_counts: dict[str, int]
    extraction_mode: str = "stream_to_disk"
    num_workers: int = 1


@dataclass(frozen=True)
class _PayloadPartTask:
    shard_index: int
    total_shards: int
    pcap_path: Path
    part_dir: Path
    payload_length: int
    max_packets_per_file: int | None
    include_empty_payload: bool
    log_every: int | None
    write_batch_size: int


@dataclass(frozen=True)
class _PayloadPartResult:
    shard_index: int
    pcap_path: str
    payload_path: Path
    metadata_path: Path
    num_packets: int
    label_counts: dict[str, int]
    protocol_counts: dict[str, int]


# Generic container folder names that are NOT traffic-class labels.
_CONTAINER_DIRS: frozenset[str] = frozenset({"", "raw", "data", "pcap", "pcaps", "dataset"})

# Remap folder names that need cleanup (folder → canonical label).
_LABEL_NORMALIZE: dict[str, str] = {
    "Benign_Final": "Benign",
}

# Strip trailing " - N" / "-N" numbering and dangling hyphens appended to stems.
_SUFFIX_RE = re.compile(r"(?:\s*-\s*\d+|-+)$")

# Keep the NPY data offset fixed so single-pass extraction can write rows first
# and patch the final row count into the header after the PCAP stream ends.
_NPY_PREAMBLE_LEN = 12
_NPY_HEADER_LEN = 4084


def _write_fixed_npy_header(handle, row_count: int, payload_length: int) -> None:
    if row_count < 0:
        raise ValueError("row_count must be >= 0")
    header = (
        "{'descr': '|u1', 'fortran_order': False, "
        f"'shape': ({int(row_count)}, {int(payload_length)}), }}"
    )
    header_bytes = header.encode("latin1")
    if len(header_bytes) + 1 > _NPY_HEADER_LEN:
        raise ValueError(f"NPY header too long for row_count={row_count}")
    padded = header_bytes + b" " * (_NPY_HEADER_LEN - len(header_bytes) - 1) + b"\n"
    handle.write(b"\x93NUMPY")
    handle.write(bytes([2, 0]))
    handle.write(_NPY_HEADER_LEN.to_bytes(4, "little"))
    handle.write(padded)


def infer_label_from_path(pcap_path: Path) -> str:
    """Derive the traffic-class label for a PCAP file.

    Two layouts are supported:

    1. Class subdir  — file lives inside a class-specific folder:
         data/raw/<ClassName>/<ClassName> - <N>.pcap
       → label = folder name  (possibly normalised via _LABEL_NORMALIZE)

    2. Flat layout   — file sits directly in a container folder (raw, data …):
         data/raw/<ClassName>.pcap
         data/raw/<ClassName> - <N>.pcap
       → label = stem with trailing " - N" stripped

    Examples:
        raw/DDoS-ACK_Fragmentation/DDoS-ACK_Fragmentation - 1.pcap  → DDoS-ACK_Fragmentation
        raw/DDoS-RSTFINFlood/DDoS-RSTFINFlood - 3.pcap              → DDoS-RSTFINFlood
        raw/Benign_Final/BenignTraffic - 1.pcap                      → Benign
        raw/Mirai-udpplain/Mirai-udpplain - 2.pcap                   → Mirai-udpplain
        raw/DDoS-SlowLoris.pcap                                       → DDoS-SlowLoris
        raw/Recon-HostDiscovery.pcap                                  → Recon-HostDiscovery
        raw/Uploading_Attack.pcap                                     → Uploading_Attack
        raw/Backdoor_Malware.pcap                                     → Backdoor_Malware
    """
    folder = pcap_path.parent.name
    if folder.lower() in _CONTAINER_DIRS:
        # Flat layout: use stem as-is (strip only trailing numbering).
        label = _SUFFIX_RE.sub("", pcap_path.stem).strip()
        return _LABEL_NORMALIZE.get(label, label)
    # Subdir layout: folder name is the ground-truth class.
    return _LABEL_NORMALIZE.get(folder, folder)


def truncate_and_pad_payload(payload: bytes, payload_length: int = 256) -> np.ndarray:
    """Convert raw payload bytes into a fixed-length uint8 vector."""
    fixed = np.zeros(payload_length, dtype=np.uint8)
    if not payload:
        return fixed
    clipped = np.frombuffer(payload[:payload_length], dtype=np.uint8)
    fixed[: clipped.shape[0]] = clipped
    return fixed


def _iter_packet_rows_scapy(
    pcap_path: Path,
    payload_length: int | None,
    max_packets: int | None,
    include_empty_payload: bool,
    log_every: int | None,
    progress_label: str | None = None,
) -> Iterator[tuple[dict[str, object], np.ndarray | None]]:
    label = infer_label_from_path(pcap_path)
    pcap_file_str = str(pcap_path)
    extracted_count = 0

    def emit_progress(seen_count: int) -> None:
        if not log_every or log_every <= 0 or seen_count % log_every != 0:
            return
        progress = "[PROGRESS]"
        if progress_label:
            progress = f"{progress}[{progress_label}]"
        print(
            f"{progress} {pcap_path.name}: seen {seen_count} packets, "
            f"extracted {extracted_count}",
            file=sys.stderr,
            flush=True,
        )

    with PcapReader(pcap_file_str) as reader:
        for packet_index, packet in enumerate(reader):
            if max_packets is not None and packet_index >= max_packets:
                break

            seen_count = packet_index + 1
            if IP not in packet:
                emit_progress(seen_count)
                continue

            ip_layer = packet[IP]
            src_port = -1
            dst_port = -1
            protocol = "OTHER"

            if TCP in packet:
                protocol = "TCP"
                src_port = int(packet[TCP].sport)
                dst_port = int(packet[TCP].dport)
            elif UDP in packet:
                protocol = "UDP"
                src_port = int(packet[UDP].sport)
                dst_port = int(packet[UDP].dport)

            raw_payload = bytes(packet[Raw].load) if Raw in packet else b""
            if not include_empty_payload and len(raw_payload) == 0:
                emit_progress(seen_count)
                continue

            payload_256 = (
                truncate_and_pad_payload(raw_payload, payload_length=payload_length)
                if payload_length is not None
                else None
            )
            extracted_count += 1
            emit_progress(seen_count)
            yield (
                {
                    "pcap_file": pcap_file_str,
                    "packet_index": packet_index,
                    "timestamp": float(getattr(packet, "time", 0.0)),
                    "label": label,
                    "src_ip": str(ip_layer.src),
                    "dst_ip": str(ip_layer.dst),
                    "src_port": src_port,
                    "dst_port": dst_port,
                    "protocol": protocol,
                    "payload_len_raw": len(raw_payload),
                },
                payload_256,
            )


def _iter_packet_rows_dpkt(
    pcap_path: Path,
    payload_length: int | None,
    max_packets: int | None,
    include_empty_payload: bool,
    log_every: int | None,
    progress_label: str | None = None,
) -> Iterator[tuple[dict[str, object], np.ndarray | None]]:
    """dpkt-based packet parser — 3–10x faster than Scapy for .pcap files.

    Falls back to Scapy only when dpkt cannot open the file at all (e.g., a .pcap
    file that is actually in pcapng format or is corrupt). If dpkt opens the file
    successfully, malformed individual packets are silently skipped rather than
    causing a fallback, so the row counts from dpkt and Scapy may differ slightly
    for corrupt captures.
    """
    label = infer_label_from_path(pcap_path)
    pcap_file_str = str(pcap_path)
    extracted_count = 0
    pl = payload_length if payload_length and payload_length > 0 else 256

    def _log(seen_count: int) -> None:
        if not log_every or log_every <= 0 or seen_count % log_every != 0:
            return
        tag = "[PROGRESS]" + (f"[{progress_label}]" if progress_label else "")
        print(
            f"{tag} {pcap_path.name}: seen {seen_count} packets, "
            f"extracted {extracted_count}",
            file=sys.stderr,
            flush=True,
        )

    dpkt_opened = False
    try:
        with open(pcap_path, "rb") as f:
            suffix = pcap_path.suffix.lower()
            if suffix == ".pcapng" and _DPKT_HAS_PCAPNG:
                pcap_iter = _dpkt_pcapng.Scanner(f)
                dlt = _DLT_EN10MB
            else:
                reader = _dpkt_pcap.Reader(f)
                dlt = reader.datalink()
                pcap_iter = reader

            dpkt_opened = True

            for packet_index, (ts, buf) in enumerate(pcap_iter):
                if max_packets is not None and packet_index >= max_packets:
                    break

                seen_count = packet_index + 1
                _log(seen_count)

                try:
                    if dlt == _DLT_EN10MB:
                        eth = _dpkt_eth.Ethernet(buf)
                        ip = eth.data
                    elif dlt in (_DLT_RAW, _DLT_RAW2):
                        ip = _dpkt_ip.IP(buf)
                    else:
                        continue
                    if not isinstance(ip, _dpkt_ip.IP):
                        continue
                except Exception:
                    continue

                transport = ip.data
                src_port = -1
                dst_port = -1
                protocol = "OTHER"
                raw_payload = b""

                if isinstance(transport, _dpkt_tcp.TCP):
                    protocol = "TCP"
                    src_port = transport.sport
                    dst_port = transport.dport
                    raw_payload = bytes(transport.data)
                elif isinstance(transport, _dpkt_udp.UDP):
                    protocol = "UDP"
                    src_port = transport.sport
                    dst_port = transport.dport
                    raw_payload = bytes(transport.data)

                if not include_empty_payload and not raw_payload:
                    continue

                if payload_length is not None:
                    vec = np.zeros(pl, dtype=np.uint8)
                    if raw_payload:
                        n = min(len(raw_payload), pl)
                        vec[:n] = np.frombuffer(raw_payload[:n], dtype=np.uint8)
                else:
                    vec = None

                extracted_count += 1
                yield (
                    {
                        "pcap_file": pcap_file_str,
                        "packet_index": packet_index,
                        "timestamp": float(ts),
                        "label": label,
                        "src_ip": _socket.inet_ntoa(ip.src),
                        "dst_ip": _socket.inet_ntoa(ip.dst),
                        "src_port": src_port,
                        "dst_port": dst_port,
                        "protocol": protocol,
                        "payload_len_raw": len(raw_payload),
                    },
                    vec,
                )
    except Exception as exc:
        if not dpkt_opened:
            print(
                f"[WARN] dpkt could not open {pcap_path.name}: {exc!r} — retrying with Scapy",
                file=sys.stderr,
                flush=True,
            )
            yield from _iter_packet_rows_scapy(
                pcap_path, payload_length, max_packets, include_empty_payload, log_every, progress_label
            )
        else:
            print(
                f"[WARN] dpkt error mid-stream in {pcap_path.name}: {exc!r} (partial extract)",
                file=sys.stderr,
                flush=True,
            )


def _iter_packet_rows(
    pcap_path: Path,
    payload_length: int | None,
    max_packets: int | None,
    include_empty_payload: bool,
    log_every: int | None,
    progress_label: str | None = None,
) -> Iterator[tuple[dict[str, object], np.ndarray | None]]:
    """Dispatch to dpkt (fast) or Scapy (fallback) based on availability and file type."""
    suffix = pcap_path.suffix.lower()
    use_dpkt = _DPKT_AVAILABLE and (
        suffix == ".pcap" or (suffix == ".pcapng" and _DPKT_HAS_PCAPNG)
    )
    if use_dpkt:
        yield from _iter_packet_rows_dpkt(
            pcap_path, payload_length, max_packets, include_empty_payload, log_every, progress_label
        )
    else:
        yield from _iter_packet_rows_scapy(
            pcap_path, payload_length, max_packets, include_empty_payload, log_every, progress_label
        )


def extract_packet_records(
    pcap_path: Path,
    payload_length: int = 256,
    max_packets: int | None = None,
    include_empty_payload: bool = False,
    log_every: int | None = None,
) -> list[PacketRecord]:
    """Extract packet metadata and fixed-size payload vectors from a PCAP file."""
    records: list[PacketRecord] = []
    for metadata_row, payload_256 in _iter_packet_rows(
        pcap_path=pcap_path,
        payload_length=payload_length,
        max_packets=max_packets,
        include_empty_payload=include_empty_payload,
        log_every=log_every,
    ):
        if payload_256 is None:
            raise RuntimeError("payload vector missing during packet extraction")
        records.append(
            PacketRecord(
                pcap_file=str(metadata_row["pcap_file"]),
                packet_index=int(metadata_row["packet_index"]),
                timestamp=float(metadata_row["timestamp"]),
                label=str(metadata_row["label"]),
                src_ip=str(metadata_row["src_ip"]),
                dst_ip=str(metadata_row["dst_ip"]),
                src_port=int(metadata_row["src_port"]),
                dst_port=int(metadata_row["dst_port"]),
                protocol=str(metadata_row["protocol"]),
                payload_len_raw=int(metadata_row["payload_len_raw"]),
                payload_256=payload_256,
            )
        )

    return records


def build_payload_dataset(
    pcap_paths: Iterable[Path],
    payload_length: int = 256,
    max_packets_per_file: int | None = None,
    include_empty_payload: bool = False,
    log_every: int | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build a matrix of payload vectors and aligned metadata rows from many PCAP files."""
    payload_rows: list[np.ndarray] = []
    metadata_rows: list[dict[str, object]] = []

    for pcap_path in pcap_paths:
        records = extract_packet_records(
            pcap_path=pcap_path,
            payload_length=payload_length,
            max_packets=max_packets_per_file,
            include_empty_payload=include_empty_payload,
            log_every=log_every,
        )

        for record in records:
            payload_rows.append(record.payload_256)
            metadata_rows.append(
                {
                    "pcap_file": record.pcap_file,
                    "packet_index": record.packet_index,
                    "timestamp": record.timestamp,
                    "label": record.label,
                    "src_ip": record.src_ip,
                    "dst_ip": record.dst_ip,
                    "src_port": record.src_port,
                    "dst_port": record.dst_port,
                    "protocol": record.protocol,
                    "payload_len_raw": record.payload_len_raw,
                }
            )

    if payload_rows:
        payload_matrix = np.stack(payload_rows, axis=0).astype(np.uint8)
    else:
        payload_matrix = np.empty((0, payload_length), dtype=np.uint8)

    metadata = pd.DataFrame(metadata_rows)
    return payload_matrix, metadata


def count_payload_dataset_rows(
    pcap_paths: Iterable[Path],
    max_packets_per_file: int | None = None,
    include_empty_payload: bool = False,
    log_every: int | None = None,
) -> tuple[int, dict[str, int], dict[str, int]]:
    """Count extractable packet rows without materializing payload vectors."""
    total_rows = 0
    label_counts: Counter[str] = Counter()
    protocol_counts: Counter[str] = Counter()

    for pcap_path in pcap_paths:
        for metadata_row, _ in _iter_packet_rows(
            pcap_path=pcap_path,
            payload_length=None,
            max_packets=max_packets_per_file,
            include_empty_payload=include_empty_payload,
            log_every=log_every,
            progress_label="count",
        ):
            total_rows += 1
            label_counts[str(metadata_row["label"])] += 1
            protocol_counts[str(metadata_row["protocol"])] += 1

    return total_rows, dict(label_counts), dict(protocol_counts)


def stream_payload_dataset_to_disk(
    pcap_paths: Iterable[Path],
    output_dir: Path,
    payload_length: int = 256,
    max_packets_per_file: int | None = None,
    include_empty_payload: bool = False,
    log_every: int | None = None,
    write_batch_size: int = 100_000,
) -> PayloadDatasetDiskResult:
    """Build payload_256.npy and metadata.csv without holding all rows in RAM.

    This performs two PCAP passes. The first pass counts rows so the NPY header
    can store the final matrix shape. The second pass writes payload rows through
    a numpy memmap and appends metadata rows directly to CSV.
    """
    if payload_length < 1:
        raise ValueError("payload_length must be >= 1")
    if write_batch_size < 1:
        raise ValueError("write_batch_size must be >= 1")

    pcap_list = list(pcap_paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / "payload_256.npy"
    metadata_path = output_dir / "metadata.csv"

    total_rows, label_counts, protocol_counts = count_payload_dataset_rows(
        pcap_paths=pcap_list,
        max_packets_per_file=max_packets_per_file,
        include_empty_payload=include_empty_payload,
        log_every=log_every,
    )

    payload_matrix = np.lib.format.open_memmap(
        str(payload_path),
        mode="w+",
        dtype=np.uint8,
        shape=(total_rows, payload_length),
    )

    row_offset = 0
    payload_batch: list[np.ndarray] = []
    metadata_batch: list[dict[str, object]] = []

    def flush_batch() -> None:
        nonlocal row_offset
        if not payload_batch:
            return
        batch_array = np.asarray(payload_batch, dtype=np.uint8)
        batch_end = row_offset + int(batch_array.shape[0])
        payload_matrix[row_offset:batch_end] = batch_array
        row_offset = batch_end
        writer.writerows(metadata_batch)
        payload_batch.clear()
        metadata_batch.clear()

    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(METADATA_COLUMNS))
        writer.writeheader()

        for pcap_path in pcap_list:
            for metadata_row, payload_256 in _iter_packet_rows(
                pcap_path=pcap_path,
                payload_length=payload_length,
                max_packets=max_packets_per_file,
                include_empty_payload=include_empty_payload,
                log_every=log_every,
                progress_label="write",
            ):
                if payload_256 is None:
                    raise RuntimeError("payload vector missing during streaming write")
                payload_batch.append(payload_256)
                metadata_batch.append(metadata_row)

                if len(payload_batch) >= write_batch_size:
                    flush_batch()

        flush_batch()

    if row_offset != total_rows:
        raise RuntimeError(f"streamed row count mismatch: wrote {row_offset}, expected {total_rows}")

    payload_matrix.flush()
    return PayloadDatasetDiskResult(
        payload_path=payload_path,
        metadata_path=metadata_path,
        num_packets=total_rows,
        label_counts=label_counts,
        protocol_counts=protocol_counts,
        extraction_mode="stream_to_disk_two_pass",
    )


def stream_payload_dataset_to_disk_single_pass(
    pcap_paths: Iterable[Path],
    output_dir: Path,
    payload_length: int = 256,
    max_packets_per_file: int | None = None,
    include_empty_payload: bool = False,
    log_every: int | None = None,
    write_batch_size: int = 100_000,
) -> PayloadDatasetDiskResult:
    """Build payload_256.npy and metadata.csv in one PCAP pass.

    The payload file is written with a fixed-size NPY header first. Payload rows
    are appended as raw uint8 bytes, then the header is patched with the final
    row count. This avoids the count pass used by numpy's open_memmap path.
    """
    if payload_length < 1:
        raise ValueError("payload_length must be >= 1")
    if write_batch_size < 1:
        raise ValueError("write_batch_size must be >= 1")

    pcap_list = list(pcap_paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / "payload_256.npy"
    metadata_path = output_dir / "metadata.csv"

    row_count = 0
    label_counts: Counter[str] = Counter()
    protocol_counts: Counter[str] = Counter()
    payload_batch: list[np.ndarray] = []
    metadata_batch: list[dict[str, object]] = []

    with payload_path.open("w+b") as payload_handle, metadata_path.open(
        "w", newline="", encoding="utf-8"
    ) as metadata_handle:
        _write_fixed_npy_header(payload_handle, row_count=0, payload_length=payload_length)
        writer = csv.DictWriter(metadata_handle, fieldnames=list(METADATA_COLUMNS))
        writer.writeheader()

        def flush_batch() -> None:
            if not payload_batch:
                return
            batch_array = np.asarray(payload_batch, dtype=np.uint8)
            batch_array.tofile(payload_handle)
            writer.writerows(metadata_batch)
            payload_batch.clear()
            metadata_batch.clear()

        for pcap_path in pcap_list:
            for metadata_row, payload_256 in _iter_packet_rows(
                pcap_path=pcap_path,
                payload_length=payload_length,
                max_packets=max_packets_per_file,
                include_empty_payload=include_empty_payload,
                log_every=log_every,
                progress_label="write",
            ):
                if payload_256 is None:
                    raise RuntimeError("payload vector missing during streaming write")

                payload_batch.append(payload_256)
                metadata_batch.append(metadata_row)
                row_count += 1
                label_counts[str(metadata_row["label"])] += 1
                protocol_counts[str(metadata_row["protocol"])] += 1

                if len(payload_batch) >= write_batch_size:
                    flush_batch()

        flush_batch()
        payload_handle.flush()
        os.fsync(payload_handle.fileno())
        payload_handle.seek(0)
        _write_fixed_npy_header(payload_handle, row_count=row_count, payload_length=payload_length)
        payload_handle.truncate(_NPY_PREAMBLE_LEN + _NPY_HEADER_LEN + row_count * payload_length)
        payload_handle.flush()
        os.fsync(payload_handle.fileno())

    return PayloadDatasetDiskResult(
        payload_path=payload_path,
        metadata_path=metadata_path,
        num_packets=row_count,
        label_counts=dict(label_counts),
        protocol_counts=dict(protocol_counts),
        extraction_mode="stream_to_disk_single_pass",
    )


def _stream_payload_part(task: _PayloadPartTask) -> _PayloadPartResult:
    payload_path = task.part_dir / f"payload_part_{task.shard_index:06d}.bin"
    metadata_path = task.part_dir / f"metadata_part_{task.shard_index:06d}.csv"

    row_count = 0
    label_counts: Counter[str] = Counter()
    protocol_counts: Counter[str] = Counter()
    payload_batch: list[np.ndarray] = []
    metadata_batch: list[dict[str, object]] = []
    progress_label = f"worker {task.shard_index + 1}/{task.total_shards}"

    with payload_path.open("wb") as payload_handle, metadata_path.open(
        "w", newline="", encoding="utf-8"
    ) as metadata_handle:
        writer = csv.DictWriter(metadata_handle, fieldnames=list(METADATA_COLUMNS))
        writer.writeheader()

        def flush_batch() -> None:
            if not payload_batch:
                return
            batch_array = np.asarray(payload_batch, dtype=np.uint8)
            batch_array.tofile(payload_handle)
            writer.writerows(metadata_batch)
            payload_batch.clear()
            metadata_batch.clear()

        for metadata_row, payload_256 in _iter_packet_rows(
            pcap_path=task.pcap_path,
            payload_length=task.payload_length,
            max_packets=task.max_packets_per_file,
            include_empty_payload=task.include_empty_payload,
            log_every=task.log_every,
            progress_label=progress_label,
        ):
            if payload_256 is None:
                raise RuntimeError("payload vector missing during parallel streaming write")

            payload_batch.append(payload_256)
            metadata_batch.append(metadata_row)
            row_count += 1
            label_counts[str(metadata_row["label"])] += 1
            protocol_counts[str(metadata_row["protocol"])] += 1

            if len(payload_batch) >= task.write_batch_size:
                flush_batch()

        flush_batch()
        payload_handle.flush()
        os.fsync(payload_handle.fileno())
        metadata_handle.flush()
        os.fsync(metadata_handle.fileno())

    return _PayloadPartResult(
        shard_index=task.shard_index,
        pcap_path=str(task.pcap_path),
        payload_path=payload_path,
        metadata_path=metadata_path,
        num_packets=row_count,
        label_counts=dict(label_counts),
        protocol_counts=dict(protocol_counts),
    )


def _resolve_worker_count(num_workers: int | None, pcap_count: int) -> int:
    if pcap_count < 1:
        return 1
    if num_workers is None or num_workers <= 0:
        return max(1, min(pcap_count, os.cpu_count() or 1))
    return max(1, min(int(num_workers), pcap_count))


def _append_metadata_rows(part_path: Path, output_handle) -> None:
    expected_header = ",".join(METADATA_COLUMNS)
    with part_path.open("r", newline="", encoding="utf-8") as part_handle:
        header = part_handle.readline().strip()
        if header != expected_header:
            raise RuntimeError(f"metadata part has unexpected header: {part_path}")
        shutil.copyfileobj(part_handle, output_handle, length=1024 * 1024)


def stream_payload_dataset_to_disk_parallel(
    pcap_paths: Iterable[Path],
    output_dir: Path,
    payload_length: int = 256,
    max_packets_per_file: int | None = None,
    include_empty_payload: bool = False,
    log_every: int | None = None,
    write_batch_size: int = 100_000,
    num_workers: int | None = None,
) -> PayloadDatasetDiskResult:
    """Build payload_256.npy and metadata.csv with one worker process per PCAP shard.

    Each worker parses one PCAP into temporary payload/metadata parts. The parent
    merges parts in the sorted input order so rows stay deterministic.
    """
    if payload_length < 1:
        raise ValueError("payload_length must be >= 1")
    if write_batch_size < 1:
        raise ValueError("write_batch_size must be >= 1")

    pcap_list = list(pcap_paths)
    worker_count = _resolve_worker_count(num_workers=num_workers, pcap_count=len(pcap_list))
    if worker_count <= 1:
        return stream_payload_dataset_to_disk_single_pass(
            pcap_paths=pcap_list,
            output_dir=output_dir,
            payload_length=payload_length,
            max_packets_per_file=max_packets_per_file,
            include_empty_payload=include_empty_payload,
            log_every=log_every,
            write_batch_size=write_batch_size,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / "payload_256.npy"
    metadata_path = output_dir / "metadata.csv"
    parts_dir = Path(tempfile.mkdtemp(prefix="_payload_parts_", dir=output_dir))
    results_by_index: dict[int, _PayloadPartResult] = {}

    try:
        tasks = [
            _PayloadPartTask(
                shard_index=idx,
                total_shards=len(pcap_list),
                pcap_path=pcap_path,
                part_dir=parts_dir,
                payload_length=payload_length,
                max_packets_per_file=max_packets_per_file,
                include_empty_payload=include_empty_payload,
                log_every=log_every,
                write_batch_size=write_batch_size,
            )
            for idx, pcap_path in enumerate(pcap_list)
        ]

        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_task = {executor.submit(_stream_payload_part, task): task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                result = future.result()
                results_by_index[result.shard_index] = result
                print(
                    f"[OK][worker {task.shard_index + 1}/{len(pcap_list)}] "
                    f"{task.pcap_path.name}: extracted {result.num_packets}",
                    file=sys.stderr,
                    flush=True,
                )

        ordered_results = [results_by_index[idx] for idx in range(len(pcap_list))]
        total_rows = sum(result.num_packets for result in ordered_results)
        label_counts: Counter[str] = Counter()
        protocol_counts: Counter[str] = Counter()
        for result in ordered_results:
            label_counts.update(result.label_counts)
            protocol_counts.update(result.protocol_counts)
            expected_bytes = int(result.num_packets) * int(payload_length)
            actual_bytes = result.payload_path.stat().st_size
            if actual_bytes != expected_bytes:
                raise RuntimeError(
                    f"payload part size mismatch for {result.payload_path}: "
                    f"{actual_bytes} != {expected_bytes}"
                )

        with payload_path.open("w+b") as payload_handle:
            _write_fixed_npy_header(payload_handle, row_count=total_rows, payload_length=payload_length)
            for result in ordered_results:
                with result.payload_path.open("rb") as part_handle:
                    shutil.copyfileobj(part_handle, payload_handle, length=16 * 1024 * 1024)
            payload_handle.truncate(_NPY_PREAMBLE_LEN + _NPY_HEADER_LEN + total_rows * payload_length)
            payload_handle.flush()
            os.fsync(payload_handle.fileno())

        with metadata_path.open("w", newline="", encoding="utf-8") as metadata_handle:
            writer = csv.DictWriter(metadata_handle, fieldnames=list(METADATA_COLUMNS))
            writer.writeheader()
            for result in ordered_results:
                _append_metadata_rows(result.metadata_path, metadata_handle)
            metadata_handle.flush()
            os.fsync(metadata_handle.fileno())

    except BaseException:
        shutil.rmtree(parts_dir, ignore_errors=True)
        raise

    shutil.rmtree(parts_dir, ignore_errors=True)
    return PayloadDatasetDiskResult(
        payload_path=payload_path,
        metadata_path=metadata_path,
        num_packets=total_rows,
        label_counts=dict(label_counts),
        protocol_counts=dict(protocol_counts),
        extraction_mode="stream_to_disk_parallel_single_pass",
        num_workers=worker_count,
    )
