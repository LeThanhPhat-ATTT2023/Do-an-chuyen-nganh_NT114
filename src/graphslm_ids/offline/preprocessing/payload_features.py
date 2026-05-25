"""Deterministic packet-payload features.

Replaces the SecureBERT-distilled CNN embedding from v1 with three deterministic
blocks that are NOT trained:

  1. Byte-distribution histogram (256-d) + entropy + printable ratio + zero fraction
  2. Hashed byte 4-gram (2048-d) via MurmurHash-style hashing (we use md5 first
     4 bytes for cheap, deterministic, dependency-free hashing)
  3. Structural HTTP-ish features (16-d): is_http_request, is_http_response,
     url_length, n_query_params, n_special_chars, content_type_textlike,
     has_multipart, header_length_est, body_length_est, n_uppercase, n_digits,
     n_alpha, n_null_bytes, n_newlines, n_tabs, max_run_repeated_byte

Total feature dim = 256 + 3 + 2048 + 16 = 2323.

The point of v2 is that no payload encoder must be trained: every dimension is
either a count, a normalized count, or a deterministic hash. That removes the
SecureBERT/student-CNN failure mode entirely (cf. design doc section 2).
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

HIST_DIM = 256
DERIVED_DIM = 3
HASH_DIM = 2048
STRUCT_DIM = 16
FEATURE_DIM = HIST_DIM + DERIVED_DIM + HASH_DIM + STRUCT_DIM  # = 2323

# Fixed offsets so callers can address feature slots without recomputation.
DERIVED_OFFSET = HIST_DIM
HASH_OFFSET = HIST_DIM + DERIVED_DIM
STRUCT_OFFSET = HIST_DIM + DERIVED_DIM + HASH_DIM

_HTTP_METHOD_RE = re.compile(rb"^(GET|POST|PUT|DELETE|HEAD|OPTIONS|PATCH) ")
_HTTP_RESPONSE_RE = re.compile(rb"^HTTP/")
_SPECIAL_CHARS = set(b"<>'\";&%=(){}[]\\")


def _byte_histogram(buf: np.ndarray, payload_len: int) -> np.ndarray:
    hist = np.bincount(buf[:payload_len], minlength=HIST_DIM).astype(np.float64)
    total = max(int(payload_len), 1)
    return (hist / total).astype(np.float32)


def _derived_from_hist(hist_norm: np.ndarray) -> np.ndarray:
    eps = 1e-12
    entropy = float(-(hist_norm * np.log2(hist_norm + eps)).sum())
    printable = float(hist_norm[0x20:0x7F].sum())
    zero_frac = float(hist_norm[0])
    return np.array([entropy, printable, zero_frac], dtype=np.float32)


def _hashed_ngram(buf: np.ndarray, payload_len: int, n: int = 4) -> np.ndarray:
    """Hash each n-byte window to one of HASH_DIM buckets and normalize."""
    out = np.zeros(HASH_DIM, dtype=np.float64)
    if payload_len < n:
        return out.astype(np.float32)
    raw = buf[:payload_len].tobytes()
    n_windows = payload_len - n + 1
    for i in range(n_windows):
        h = hashlib.md5(raw[i : i + n]).digest()
        bucket = int.from_bytes(h[:4], "big") % HASH_DIM
        out[bucket] += 1.0
    if n_windows > 0:
        out /= n_windows
    return out.astype(np.float32)


def _http_structural(buf: np.ndarray, payload_len: int) -> np.ndarray:
    """Hand-curated 16 deterministic features capturing HTTP/textual structure."""
    raw = buf[: min(payload_len, 256)].tobytes()
    if not raw:
        return np.zeros(STRUCT_DIM, dtype=np.float32)

    is_req = 1.0 if _HTTP_METHOD_RE.match(raw) else 0.0
    is_resp = 1.0 if _HTTP_RESPONSE_RE.match(raw) else 0.0
    url_length = 0.0
    n_query_params = 0.0
    if is_req:
        # Parse the start line: "<METHOD> <URL> HTTP/x.y\r\n"
        try:
            first_line = raw.split(b"\r\n", 1)[0]
            parts = first_line.split(b" ")
            if len(parts) >= 2:
                url = parts[1]
                url_length = float(len(url))
                if b"?" in url:
                    n_query_params = float(url.count(b"&") + 1)
        except Exception:
            pass
    n_special = float(sum(1 for b in raw if b in _SPECIAL_CHARS))
    content_type_textlike = 0.0
    if b"text/" in raw.lower() or b"application/json" in raw.lower():
        content_type_textlike = 1.0
    has_multipart = 1.0 if b"multipart/form-data" in raw.lower() else 0.0
    # Header / body split estimate (works for HTTP-like text; zero for binary).
    if b"\r\n\r\n" in raw:
        header_part, body_part = raw.split(b"\r\n\r\n", 1)
        header_length_est = float(len(header_part))
        body_length_est = float(len(body_part))
    else:
        header_length_est = float(len(raw))
        body_length_est = 0.0
    n_uppercase = float(sum(1 for b in raw if 0x41 <= b <= 0x5A))
    n_digits = float(sum(1 for b in raw if 0x30 <= b <= 0x39))
    n_alpha = float(
        sum(1 for b in raw if (0x41 <= b <= 0x5A) or (0x61 <= b <= 0x7A))
    )
    n_null_bytes = float(raw.count(b"\x00"))
    n_newlines = float(raw.count(b"\n"))
    n_tabs = float(raw.count(b"\t"))
    # Cheapest run-length: longest stretch of identical bytes.
    max_run = 0
    cur_run = 1
    for i in range(1, len(raw)):
        if raw[i] == raw[i - 1]:
            cur_run += 1
        else:
            if cur_run > max_run:
                max_run = cur_run
            cur_run = 1
    if cur_run > max_run:
        max_run = cur_run

    return np.array(
        [
            is_req,
            is_resp,
            url_length,
            n_query_params,
            n_special,
            content_type_textlike,
            has_multipart,
            header_length_est,
            body_length_est,
            n_uppercase,
            n_digits,
            n_alpha,
            n_null_bytes,
            n_newlines,
            n_tabs,
            float(max_run),
        ],
        dtype=np.float32,
    )


def compute_packet_payload_features(
    payload_bytes: bytes | np.ndarray, payload_len: int
) -> np.ndarray:
    """Return a float32 vector of length :data:`FEATURE_DIM` for one packet."""
    if isinstance(payload_bytes, bytes):
        buf = np.frombuffer(payload_bytes, dtype=np.uint8)
    else:
        buf = np.asarray(payload_bytes, dtype=np.uint8).ravel()
    pl = int(max(0, min(payload_len, buf.shape[0])))

    hist_norm = _byte_histogram(buf, pl) if pl > 0 else np.zeros(HIST_DIM, dtype=np.float32)
    derived = _derived_from_hist(hist_norm.astype(np.float64))
    ngram = _hashed_ngram(buf, pl)
    struct = _http_structural(buf, pl)

    out = np.empty(FEATURE_DIM, dtype=np.float32)
    out[:HIST_DIM] = hist_norm
    out[DERIVED_OFFSET:HASH_OFFSET] = derived
    out[HASH_OFFSET:STRUCT_OFFSET] = ngram
    out[STRUCT_OFFSET:] = struct
    return out


def aggregate_flow_payload_features(
    payload_matrix: np.ndarray, packet_payload_idx: dict[str, np.ndarray]
) -> tuple[np.ndarray, list[str]]:
    """Mean per-packet feature vector over each flow's packets.

    Args:
      payload_matrix: ``(N_packets_total, payload_length)`` uint8 array.
      packet_payload_idx: ``flow_id -> int64 array of row indices into payload_matrix``.

    Returns ``(M, FEATURE_NAMES)`` where ``M`` has shape ``(N_flows, FEATURE_DIM)``.
    Flows with zero packets get a zero vector (no NaNs propagate downstream).
    """
    flow_ids = list(packet_payload_idx.keys())
    out = np.zeros((len(flow_ids), FEATURE_DIM), dtype=np.float32)
    payload_length = int(payload_matrix.shape[1])
    for row, fid in enumerate(flow_ids):
        idx = np.asarray(packet_payload_idx[fid], dtype=np.int64)
        if idx.size == 0:
            continue
        feats_stack = np.zeros((idx.size, FEATURE_DIM), dtype=np.float32)
        for k, pkt_row in enumerate(idx):
            raw = np.asarray(payload_matrix[pkt_row], dtype=np.uint8)
            # actual payload length is bounded by the matrix width and by the
            # number of non-zero bytes in the row; the matrix is already padded,
            # so we treat the whole row as the payload window.
            feats_stack[k] = compute_packet_payload_features(raw, payload_length)
        out[row] = feats_stack.mean(axis=0)
    return out, _feature_names()


def _feature_names() -> list[str]:
    """Stable ordered feature name list (used by the graph artifact meta)."""
    names: list[str] = []
    names += [f"byte_{i:03d}" for i in range(HIST_DIM)]
    names += ["pl_entropy", "pl_printable", "pl_zerofrac"]
    names += [f"hng_{i:04d}" for i in range(HASH_DIM)]
    names += [
        "is_http_request",
        "is_http_response",
        "url_length",
        "n_query_params",
        "n_special_chars",
        "content_type_textlike",
        "has_multipart",
        "header_length_est",
        "body_length_est",
        "n_uppercase",
        "n_digits",
        "n_alpha",
        "n_null_bytes",
        "n_newlines",
        "n_tabs",
        "max_run_repeated_byte",
    ]
    return names


FEATURE_NAMES = _feature_names()
