from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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


def infer_label_from_path(pcap_path: Path) -> str:
    """Infer a coarse label from a file name prefix."""
    stem = pcap_path.stem
    if "-" in stem:
        return stem.split("-")[0]
    if "_" in stem:
        return stem.split("_")[0]
    return stem


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
) -> list[PacketRecord]:
    """Extract packet metadata and fixed-size payload vectors from a PCAP file."""
    label = infer_label_from_path(pcap_path)
    records: list[PacketRecord] = []

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

    return records


def build_payload_dataset(
    pcap_paths: Iterable[Path],
    payload_length: int = 256,
    max_packets_per_file: int | None = None,
    include_empty_payload: bool = False,
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
