"""Unit tests for the v3 payload tokenizer.

Tokenizer guarantees we lock in:

* Empty input → empty output (used by every callsite that filters
  zero-payload packets).
* Determinism (PMI pipeline assumes identical input → identical token set).
* Per-prefix output shape (``b4:``, ``b8:``, ``t:``) on canonical inputs.
* 200-token cap honored on large payloads.
"""
from __future__ import annotations

from graphslm_ids.offline.preprocessing.tokenizer import (
    BYTE4_PREFIX,
    BYTE8_PREFIX,
    TEXT_PREFIX,
    is_mostly_printable,
    tokenize_payload,
)


def test_empty_payload_returns_empty_set() -> None:
    assert tokenize_payload(b"") == set()


def test_determinism_same_payload_twice() -> None:
    payload = b"GET /index.html HTTP/1.1\r\n\r\n"
    assert tokenize_payload(payload) == tokenize_payload(payload)


def test_http_request_emits_expected_text_tokens() -> None:
    """HTTP request payload should expose URI-segment and header tokens."""
    tokens = tokenize_payload(
        b"GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n"
    )
    # Tokenizer lowercases before emitting t:; tokens are URI/header pieces.
    assert TEXT_PREFIX + "get" in tokens
    assert TEXT_PREFIX + "index.html" in tokens
    assert TEXT_PREFIX + "host:" in tokens


def test_binary_garbage_has_no_text_tokens() -> None:
    """A binary buffer below the printable threshold yields only b4/b8 tokens.

    Use 64 bytes so b4 (61 grams) + b8 (15 stride-4 grams) both fit under the
    200-token cap; otherwise b4 saturates the cap and b8 never runs.
    """
    payload = bytes(range(64))  # printable ratio < 0.7, small enough for b8 to fit
    tokens = tokenize_payload(payload)
    assert all(not t.startswith(TEXT_PREFIX) for t in tokens)
    assert any(t.startswith(BYTE4_PREFIX) for t in tokens)
    assert any(t.startswith(BYTE8_PREFIX) for t in tokens)


def test_token_cap_honored_on_large_payload() -> None:
    """A 10K-byte payload must not blow past the 200-token-per-packet cap."""
    payload = b"A" * 10_000
    tokens = tokenize_payload(payload)
    assert len(tokens) <= 200


def test_all_documented_prefixes_present_on_mixed_payload() -> None:
    """A mixed-content printable payload exercises all three prefixes."""
    # Long enough to produce b8 grams (stride 4 needs 8+ bytes).
    payload = b"username=admin&password=letmein&token=abcdef0123456789"
    tokens = tokenize_payload(payload)
    assert any(t.startswith(BYTE4_PREFIX) for t in tokens)
    assert any(t.startswith(BYTE8_PREFIX) for t in tokens)
    assert any(t.startswith(TEXT_PREFIX) for t in tokens)


def test_is_mostly_printable_empty_is_false() -> None:
    # Module docstring says empty buffers are NOT printable; lock the contract.
    assert is_mostly_printable(b"") is False


def test_tokenizer_handles_all_fixture_payloads(tiny_payloads: list[bytes]) -> None:
    """Smoke-test: every fixture payload tokenises without raising."""
    for payload in tiny_payloads:
        tokens = tokenize_payload(payload)
        assert isinstance(tokens, set)
