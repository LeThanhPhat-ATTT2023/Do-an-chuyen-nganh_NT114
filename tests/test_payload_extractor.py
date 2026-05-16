import csv
from pathlib import Path

import numpy as np
from scapy.layers.inet import IP, TCP, UDP
from scapy.packet import Raw
from scapy.utils import PcapWriter

from graphslm_ids.offline.preprocessing.pcap_payload_extractor import (
    infer_label_from_path,
    stream_payload_dataset_to_disk_parallel,
    truncate_and_pad_payload,
)


def test_truncate_and_pad_payload_short_sequence() -> None:
    payload = bytes([1, 2, 3])
    out = truncate_and_pad_payload(payload, payload_length=8)
    assert out.dtype == np.uint8
    assert out.shape == (8,)
    assert out.tolist() == [1, 2, 3, 0, 0, 0, 0, 0]


def test_truncate_and_pad_payload_clips_long_sequence() -> None:
    payload = bytes(range(10))
    out = truncate_and_pad_payload(payload, payload_length=4)
    assert out.tolist() == [0, 1, 2, 3]


def test_infer_label_from_path_normalizes_flat_layout_suffixes() -> None:
    assert infer_label_from_path(Path("data/raw/DDoS-HTTP_Flood-.pcap")) == "DDoS-HTTP_Flood"
    assert infer_label_from_path(Path("data/raw/DDoS-HTTP_Flood - 1.pcap")) == "DDoS-HTTP_Flood"


def _write_pcap(path: Path, packets: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with PcapWriter(str(path), sync=True) as writer:
        for packet in packets:
            writer.write(packet)


def test_parallel_stream_to_disk_merges_pcap_parts(tmp_path: Path) -> None:
    first_pcap = tmp_path / "raw" / "Alpha.pcap"
    second_pcap = tmp_path / "raw" / "Beta.pcap"
    _write_pcap(
        first_pcap,
        [
            IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=1111, dport=80) / Raw(load=bytes([1, 2])),
            IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=1111, dport=80),
            IP(src="10.0.0.3", dst="10.0.0.4") / UDP(sport=2222, dport=53) / Raw(load=bytes([3, 4, 5, 6])),
        ],
    )
    _write_pcap(
        second_pcap,
        [
            IP(src="10.0.1.1", dst="10.0.1.2") / TCP(sport=3333, dport=443) / Raw(load=bytes([9])),
        ],
    )

    result = stream_payload_dataset_to_disk_parallel(
        pcap_paths=[first_pcap, second_pcap],
        output_dir=tmp_path / "out",
        payload_length=4,
        write_batch_size=1,
        num_workers=2,
    )

    payload = np.load(result.payload_path)
    assert result.num_workers == 2
    assert result.num_packets == 3
    assert payload.tolist() == [[1, 2, 0, 0], [3, 4, 5, 6], [9, 0, 0, 0]]

    with result.metadata_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["label"] for row in rows] == ["Alpha", "Alpha", "Beta"]
    assert [row["protocol"] for row in rows] == ["TCP", "UDP", "TCP"]
