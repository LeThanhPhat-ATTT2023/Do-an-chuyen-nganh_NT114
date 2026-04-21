from pathlib import Path

import numpy as np

from graphslm_ids.data.pcap_payload_extractor import infer_label_from_path, truncate_and_pad_payload


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


def test_infer_label_from_path_uses_prefix() -> None:
    label = infer_label_from_path(Path("Mirai-UDPFlood-001.pcap"))
    assert label == "Mirai"
