from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from scapy.layers.inet import IP, TCP, UDP
from scapy.packet import Raw
from scapy.utils import PcapReader


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


# Generic container folder names that are NOT traffic-class labels.
_CONTAINER_DIRS: frozenset[str] = frozenset({"raw", "data", "pcap", "pcaps", "dataset"})

# Remap folder names that need cleanup (folder → canonical label).
_LABEL_NORMALIZE: dict[str, str] = {
    "Benign_Final": "Benign",
}

# Strip trailing " - N" or "-N" numbering appended to multi-file stems.
_SUFFIX_RE = re.compile(r"\s*-\s*\d+$")


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


def extract_packet_records(
    pcap_path: Path,
    payload_length: int = 256,
    max_packets: int | None = None,
    include_empty_payload: bool = False,
    log_every: int | None = None,
) -> list[PacketRecord]:
    """Extract packet metadata and fixed-size payload vectors from a PCAP file."""
    label = infer_label_from_path(pcap_path)
    records: list[PacketRecord] = []
    extracted_count = 0

    with PcapReader(str(pcap_path)) as reader:
        for packet_index, packet in enumerate(reader):
            if max_packets is not None and packet_index >= max_packets:
                break

            if IP not in packet:
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
                continue

            records.append(
                PacketRecord(
                    pcap_file=str(pcap_path),
                    packet_index=packet_index,
                    timestamp=float(getattr(packet, "time", 0.0)),
                    label=label,
                    src_ip=str(ip_layer.src),
                    dst_ip=str(ip_layer.dst),
                    src_port=src_port,
                    dst_port=dst_port,
                    protocol=protocol,
                    payload_len_raw=len(raw_payload),
                    payload_256=truncate_and_pad_payload(raw_payload, payload_length=payload_length),
                )
            )
            extracted_count += 1

            if log_every and log_every > 0 and (packet_index + 1) % log_every == 0:
                print(
                    f"[PROGRESS] {pcap_path.name}: seen {packet_index + 1} packets, "
                    f"extracted {extracted_count}",
                    file=sys.stderr,
                    flush=True,
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
