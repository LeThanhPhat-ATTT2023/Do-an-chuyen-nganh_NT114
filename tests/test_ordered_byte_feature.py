"""Tests for the deterministic ORDERED-BYTE block (EG-HGT v5, Phase 1).

The ordered-byte block appends ``buf[:K] / 255.0`` (K = ORDERED_DIM, in byte
order, zero-padded) to the per-packet feature vector. This is the position-
preserving signal GNN4ID exploits — fully deterministic (a divide + pad, no
learned parameters, no hashing, no randomness).

These tests are the spec for the new block:
  (a) determinism — same input -> identical output across two calls.
  (b) output shape == FEATURE_DIM and dtype is float.
  (c) byte/255 correctness on a known byte sequence.
  (d) zero-padding — short payloads leave the tail of the ordered block at 0.0.
  (e) the existing histogram / hashed-ngram / derived / struct blocks are
      unchanged (their offsets and values are preserved).
"""
from __future__ import annotations

import numpy as np

from graphslm_ids.offline.preprocessing.payload_features import (
    DERIVED_OFFSET,
    FEATURE_DIM,
    FEATURE_NAMES,
    HASH_DIM,
    HASH_OFFSET,
    HIST_DIM,
    ORDERED_DIM,
    ORDERED_OFFSET,
    STRUCT_DIM,
    STRUCT_OFFSET,
    compute_packet_payload_features,
)

# --- Independent reference implementation of the ordered-byte block. ---------
# Deliberately reimplemented here so the test does not just mirror the SUT.


def _expected_ordered(raw: bytes, k: int) -> np.ndarray:
    out = np.zeros(k, dtype=np.float32)
    take = min(len(raw), k)
    for i in range(take):
        out[i] = raw[i] / 255.0
    return out


# --- Dimension wiring --------------------------------------------------------


def test_feature_dim_includes_ordered_block() -> None:
    assert ORDERED_DIM == 256
    # FEATURE_DIM must account for every block exactly once.
    assert FEATURE_DIM == HIST_DIM + 3 + HASH_DIM + STRUCT_DIM + ORDERED_DIM
    # The ordered block must occupy the final ORDERED_DIM slots.
    assert ORDERED_OFFSET == FEATURE_DIM - ORDERED_DIM
    # Offsets must be strictly ordered and non-overlapping.
    assert DERIVED_OFFSET < HASH_OFFSET < STRUCT_OFFSET < ORDERED_OFFSET


def test_feature_names_cover_ordered_block() -> None:
    assert len(FEATURE_NAMES) == FEATURE_DIM
    assert len(set(FEATURE_NAMES)) == FEATURE_DIM
    # The names for the ordered block are contiguous at the tail.
    tail = FEATURE_NAMES[ORDERED_OFFSET:]
    assert len(tail) == ORDERED_DIM
    assert all(name.startswith("obyte_") for name in tail)


# --- (a) determinism ---------------------------------------------------------


def test_ordered_block_is_deterministic() -> None:
    payload = b"GET /?id=1' OR 1=1-- HTTP/1.1\r\n\r\n" + bytes(range(0, 200))
    v1 = compute_packet_payload_features(payload, len(payload))
    v2 = compute_packet_payload_features(payload, len(payload))
    assert np.array_equal(v1, v2), "whole vector must be deterministic"
    o1 = v1[ORDERED_OFFSET:]
    o2 = v2[ORDERED_OFFSET:]
    assert np.array_equal(o1, o2), "ordered block must be deterministic"


# --- (b) shape + dtype -------------------------------------------------------


def test_output_shape_and_float_dtype() -> None:
    v = compute_packet_payload_features(bytes([1, 2, 3, 4]), 4)
    assert v.shape == (FEATURE_DIM,)
    assert np.issubdtype(v.dtype, np.floating)
    ordered = v[ORDERED_OFFSET:]
    assert ordered.shape == (ORDERED_DIM,)


# --- (c) byte / 255 correctness ----------------------------------------------


def test_byte_over_255_correctness_known_sequence() -> None:
    payload = bytes([0, 128, 255, 1, 254, 64])
    v = compute_packet_payload_features(payload, len(payload))
    ordered = v[ORDERED_OFFSET:]
    expected_head = np.array(
        [0.0, 128 / 255.0, 1.0, 1 / 255.0, 254 / 255.0, 64 / 255.0],
        dtype=np.float32,
    )
    assert np.allclose(ordered[: len(payload)], expected_head)
    # Boundary values are exact.
    assert ordered[0] == 0.0
    assert ordered[2] == 1.0


def test_ordered_block_matches_independent_reference() -> None:
    payload = bytes((i * 37 + 11) % 256 for i in range(300))  # longer than K
    v = compute_packet_payload_features(payload, len(payload))
    ordered = v[ORDERED_OFFSET:]
    expected = _expected_ordered(payload, ORDERED_DIM)
    assert np.array_equal(ordered, expected)


def test_ordered_block_truncates_at_K() -> None:
    # Payload longer than K: only the first K bytes appear, in order.
    payload = bytes([7]) * (ORDERED_DIM + 50)
    v = compute_packet_payload_features(payload, len(payload))
    ordered = v[ORDERED_OFFSET:]
    assert np.allclose(ordered, np.full(ORDERED_DIM, 7 / 255.0, dtype=np.float32))
    assert ordered.shape == (ORDERED_DIM,)


# --- (d) zero-padding --------------------------------------------------------


def test_zero_padding_for_short_payload() -> None:
    payload = bytes([10, 20, 30])  # much shorter than K
    v = compute_packet_payload_features(payload, len(payload))
    ordered = v[ORDERED_OFFSET:]
    expected_head = np.array([10 / 255.0, 20 / 255.0, 30 / 255.0], dtype=np.float32)
    assert np.allclose(ordered[:3], expected_head)
    # Everything after the payload must be exactly 0.0.
    assert np.all(ordered[3:] == 0.0)


def test_empty_payload_gives_all_zero_ordered_block() -> None:
    v = compute_packet_payload_features(b"", 0)
    ordered = v[ORDERED_OFFSET:]
    assert np.all(ordered == 0.0)


def test_payload_len_smaller_than_buffer_is_respected() -> None:
    # buf has 10 bytes but payload_len says only 4 are real -> bytes 4..9 are
    # NOT part of the payload, so the ordered tail past index 3 must be zero.
    buf = bytes([1, 2, 3, 4, 99, 99, 99, 99, 99, 99])
    v = compute_packet_payload_features(buf, 4)
    ordered = v[ORDERED_OFFSET:]
    expected_head = np.array(
        [1 / 255.0, 2 / 255.0, 3 / 255.0, 4 / 255.0], dtype=np.float32
    )
    assert np.allclose(ordered[:4], expected_head)
    assert np.all(ordered[4:] == 0.0)


# --- (e) existing blocks unchanged -------------------------------------------


def test_existing_blocks_unchanged_values() -> None:
    """The histogram / derived / hashed-ngram / struct blocks must be byte-for-
    byte what they were before the ordered block existed.

    We pin concrete expectations computed from the pre-existing definitions
    (independent of the SUT) so that a regression in those blocks is caught even
    though the ordered block was appended.
    """
    payload = b"GET /a HTTP/1.1\r\n\r\n"
    raw = np.frombuffer(payload, dtype=np.uint8)
    pl = len(payload)
    v = compute_packet_payload_features(payload, pl)

    # Histogram block: normalized byte counts over the payload length.
    expected_hist = (
        np.bincount(raw, minlength=HIST_DIM).astype(np.float64) / pl
    ).astype(np.float32)
    assert np.allclose(v[:HIST_DIM], expected_hist)

    # Derived block: entropy, printable ratio, zero fraction (from hist).
    eps = 1e-12
    hist_norm = expected_hist.astype(np.float64)
    entropy = float(-(hist_norm * np.log2(hist_norm + eps)).sum())
    printable = float(hist_norm[0x20:0x7F].sum())
    zero_frac = float(hist_norm[0])
    expected_derived = np.array([entropy, printable, zero_frac], dtype=np.float32)
    assert np.allclose(v[DERIVED_OFFSET:HASH_OFFSET], expected_derived)

    # Struct block: is_http_request flag is set for an HTTP request payload.
    struct = v[STRUCT_OFFSET:ORDERED_OFFSET]
    assert struct.shape == (STRUCT_DIM,)
    assert struct[0] == 1.0  # is_http_request

    # Hashed-ngram block is non-empty for a >=4-byte payload (sums to ~1.0).
    hashed = v[HASH_OFFSET:STRUCT_OFFSET]
    assert hashed.shape == (HASH_DIM,)
    assert np.isclose(hashed.sum(), 1.0, atol=1e-5)


def test_existing_block_offsets_are_stable() -> None:
    # The ordered block is APPENDED, so the legacy offsets are unchanged.
    assert HIST_DIM == 256
    assert DERIVED_OFFSET == 256
    assert HASH_OFFSET == 256 + 3
    assert STRUCT_OFFSET == 256 + 3 + HASH_DIM
