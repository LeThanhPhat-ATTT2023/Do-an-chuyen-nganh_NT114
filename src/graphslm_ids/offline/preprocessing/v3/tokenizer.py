"""Deterministic packet-payload tokenizer for the v3 PMI Ensemble.

Produces three classes of tokens per payload (set semantics — duplicates within
one packet are not counted twice):

* ``b4:HEX`` byte 4-grams, sliding window over all positions. Designed to
  capture short magic / opcode sequences that don't lie on word boundaries.
* ``b8:HEX`` byte 8-grams with stride 4 (50% overlap). Captures longer patterns
  while halving the token count vs sliding window.
* ``t:LOWER`` HTTP/text tokens, only emitted when the payload looks mostly
  printable (printable byte ratio > 0.7). Captures structured web attacks
  (URL query keys, header names) that the byte n-grams would shred into
  uninterpretable hex.

Two design choices that look strange but are deliberate:

* **Set semantics, not multiset.** PMI is computed over per-packet token
  presence, so repeated tokens in the same packet inflate nothing. Returning a
  set forces this contract at the API boundary.
* **Per-packet 200-token cap.** Real IoT-2023 payloads can be megabytes (RTSP
  streams, file uploads). Without a cap the b4/b8 grams alone push a single
  packet to ~1M tokens and OOM the tokenize-200K-packets stage. 200 is enough
  to characterise the *beginning* of a payload, which is where protocol
  headers and attack signatures live.

Determinism: identical input bytes -> identical output set. No RNG, no system
clock, no hashing collisions (we use ``bytes.hex()`` which is bijective).

Memory budget: ~3 KB per packet (200 tokens * ~16 chars/token). For 200K
sampled packets that's ~600 MB peak when held as a Python list of sets — well
within the 1.5 GB budget allocated to stage 4a.
"""
from __future__ import annotations

import re

BYTE4_PREFIX = "b4:"
BYTE8_PREFIX = "b8:"
TEXT_PREFIX = "t:"

# Cap on tokens per packet. Keeps the first 200 tokens in generation order
# (b4 first, then b8, then t:) so payload-prefix evidence is preserved even
# when the body is huge.
_MAX_TOKENS_PER_PACKET = 200

# Split text tokens on web-protocol delimiters. Keep the regex precompiled so
# tokenisation of a 200K-packet subsample doesn't recompile per call.
_TEXT_SPLIT_RE = re.compile(r"[ \t\r\n?&=/]+")

# Printable byte range: ASCII 0x20..0x7E (space..tilde) plus tab/newline/CR.
# Anything else is binary noise and the text-tokeniser would emit garbage.
_PRINTABLE_SET = frozenset(b" \t\r\n" + bytes(range(0x20, 0x7F)))


def is_mostly_printable(raw: bytes, threshold: float = 0.7) -> bool:
    """Return True iff the printable-byte ratio of ``raw`` exceeds ``threshold``.

    Empty buffers are considered NOT printable: there is no text to tokenise.
    """
    if not raw:
        return False
    printable = sum(1 for b in raw if b in _PRINTABLE_SET)
    return (printable / len(raw)) > threshold


def tokenize_payload(raw: bytes) -> set[str]:
    """Tokenise a packet payload into the v3 hybrid token alphabet.

    Args:
        raw: raw payload bytes. May be empty.

    Returns:
        A set of token strings, capped at 200 tokens per packet (insertion
        order is b4 grams -> b8 grams -> text tokens; first 200 retained).
        Empty payload -> empty set.
    """
    if not raw:
        return set()

    tokens: list[str] = []
    seen: set[str] = set()

    def _add(tok: str) -> bool:
        """Return False once the cap is hit so callers can short-circuit."""
        if tok in seen:
            return True
        if len(seen) >= _MAX_TOKENS_PER_PACKET:
            return False
        seen.add(tok)
        tokens.append(tok)
        return True

    # b4: sliding 4-grams over all positions. Cheap and dense.
    n = len(raw)
    for i in range(n - 3):
        if not _add(BYTE4_PREFIX + raw[i : i + 4].hex()):
            break

    # b8: stride-4 8-grams (50% overlap with itself, not the b4 grams).
    if len(seen) < _MAX_TOKENS_PER_PACKET:
        for i in range(0, n - 7, 4):
            if not _add(BYTE8_PREFIX + raw[i : i + 8].hex()):
                break

    # t: HTTP/text tokens, only on mostly-printable payloads.
    if len(seen) < _MAX_TOKENS_PER_PACKET and is_mostly_printable(raw):
        # latin-1 errors='ignore' is lossless on the printable range and never
        # raises — safe even if a binary byte sneaks through.
        text = raw.decode("latin-1", errors="ignore").lower()
        for piece in _TEXT_SPLIT_RE.split(text):
            if 2 <= len(piece) <= 64:
                if not _add(TEXT_PREFIX + piece):
                    break

    return seen
