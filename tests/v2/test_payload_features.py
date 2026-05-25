"""Tests for the deterministic v2 payload feature builder."""
from __future__ import annotations

import numpy as np

from graphslm_ids.offline.preprocessing.v2.payload_features import (
    FEATURE_DIM,
    FEATURE_NAMES,
    HASH_OFFSET,
    STRUCT_OFFSET,
    aggregate_flow_payload_features,
    compute_packet_payload_features,
)


def test_packet_features_shape_and_determinism() -> None:
    payload = b"GET /?id=1' OR 1=1-- HTTP/1.1\r\n\r\n"
    v1 = compute_packet_payload_features(payload, len(payload))
    v2 = compute_packet_payload_features(payload, len(payload))
    assert v1.shape == (FEATURE_DIM,)
    assert np.array_equal(v1, v2), "feature extraction must be deterministic"


def test_http_request_flag_set_for_http_payload() -> None:
    http = compute_packet_payload_features(b"GET /a HTTP/1.1\r\n\r\n", 19)
    binary = compute_packet_payload_features(bytes([0x00, 0xFF] * 16), 32)
    # is_http_request is the first slot of the STRUCT block.
    assert http[STRUCT_OFFSET] == 1.0
    assert binary[STRUCT_OFFSET] == 0.0


def test_hash_block_signals_distinct_payloads() -> None:
    a = compute_packet_payload_features(b"SELECT * FROM users WHERE 1=1", 30)
    b = compute_packet_payload_features(b"image/jpeg\x00\xff\xd8\xff\xe0\xff\xff", 18)
    # The hashed-ngram block (slice between HASH_OFFSET and STRUCT_OFFSET) must
    # differ between a text payload and a binary one — otherwise the hash is
    # useless. The byte-histogram block almost certainly differs too.
    a_hash = a[HASH_OFFSET:STRUCT_OFFSET]
    b_hash = b[HASH_OFFSET:STRUCT_OFFSET]
    assert not np.allclose(a_hash, b_hash)


def test_aggregate_flow_payload_features_shapes_and_names() -> None:
    payload_matrix = np.zeros((4, 32), dtype=np.uint8)
    payload_matrix[1, :3] = list(b"GET")
    payload_matrix[3, :10] = list(b"<script>al")
    pkt_idx = {
        "f0": np.array([0, 1], dtype=np.int64),
        "f1": np.array([2, 3], dtype=np.int64),
        "f2": np.array([], dtype=np.int64),
    }
    M, names = aggregate_flow_payload_features(payload_matrix, pkt_idx)
    assert M.shape == (3, FEATURE_DIM)
    assert names == FEATURE_NAMES
    # Empty flow -> zero row, no NaN.
    assert np.all(M[2] == 0.0)
    assert not np.isnan(M).any()


def test_feature_names_unique_and_full() -> None:
    assert len(FEATURE_NAMES) == FEATURE_DIM
    assert len(set(FEATURE_NAMES)) == FEATURE_DIM
