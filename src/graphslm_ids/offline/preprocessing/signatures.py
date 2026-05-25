"""Apply signatures to payload bytes or flow rows and return MITRE technique hits.

The split between this module and :mod:`._signature_rules` is intentional: the
rules data lives separately so the corpus can be edited / extended without
touching the application code, and so unit tests can reason about both halves
independently.
"""
from __future__ import annotations

import base64
import re
import urllib.parse
from pathlib import Path

import pandas as pd

from graphslm_ids.offline.preprocessing._signature_rules import (
    ALL_REFERENCED_TECHNIQUES,
    FLOW_SIGNATURES,
    PAYLOAD_SIGNATURES,
)

_HTML_ENTITY_RE = re.compile(rb"&(#x?[0-9a-fA-F]+|amp|lt|gt|quot|apos);")
_HTML_ENTITY_MAP = {
    b"&amp;": b"&",
    b"&lt;": b"<",
    b"&gt;": b">",
    b"&quot;": b'"',
    b"&apos;": b"'",
}


def _html_entity_decode(buf: bytes) -> bytes:
    """Naive HTML entity decoder for the named entities and numeric refs.

    We don't pull in html.parser for this — payloads are short and the regex
    + dict cover the cases that matter for WAF-style detection.
    """
    out = buf
    for entity, replacement in _HTML_ENTITY_MAP.items():
        out = out.replace(entity, replacement)

    def _num_sub(match: re.Match[bytes]) -> bytes:
        body = match.group(1)
        try:
            if body.startswith(b"#x") or body.startswith(b"#X"):
                return bytes([int(body[2:], 16) & 0xFF])
            if body.startswith(b"#"):
                return bytes([int(body[1:]) & 0xFF])
        except Exception:
            return match.group(0)
        return match.group(0)

    out = _HTML_ENTITY_RE.sub(_num_sub, out)
    return out


def _transform(payload: bytes, ops: tuple[str, ...]) -> bytes:
    """Apply transformation pipeline (deterministic, best-effort)."""
    out = payload
    for op in ops:
        if op == "lowercase":
            out = out.lower()
        elif op == "url_decode":
            try:
                out = urllib.parse.unquote_to_bytes(out)
            except Exception:
                pass
        elif op == "html_entity_decode":
            out = _html_entity_decode(out)
        elif op == "base64_decode":
            try:
                # Only attempt decode if it plausibly looks like base64 (no
                # spaces, only base64 alphabet). Otherwise it produces garbage
                # and pollutes downstream matching.
                stripped = bytes(b for b in out if not (b in b"\r\n\t "))
                if stripped and len(stripped) % 4 == 0:
                    out = base64.b64decode(stripped, validate=False)
            except Exception:
                pass
    return out


def match_payload_signatures(payload: bytes) -> list[tuple[str, float]]:
    """Return a deduped list of ``(technique_id, weight)`` for matched rules.

    Weights for the same technique are summed and clipped to 1.0 (so multiple
    pieces of evidence reinforce confidence without runaway).
    """
    if not payload:
        return []
    hits: dict[str, float] = {}
    for rule in PAYLOAD_SIGNATURES:
        transformed = _transform(payload, rule.transformations)
        if re.search(rule.regex.encode("latin-1"), transformed, flags=re.IGNORECASE):
            hits[rule.technique_id] = min(
                1.0, hits.get(rule.technique_id, 0.0) + rule.weight
            )
    return list(hits.items())


def match_flow_signatures(flow_row) -> list[tuple[str, float]]:
    """Same shape as :func:`match_payload_signatures`, but for a flow features row."""
    hits: dict[str, float] = {}
    for rule in FLOW_SIGNATURES:
        try:
            weight = float(rule.condition_fn(flow_row))
        except Exception:
            weight = 0.0
        if weight > 0:
            hits[rule.technique_id] = min(
                1.0, hits.get(rule.technique_id, 0.0) + weight
            )
    return list(hits.items())


def validate_techniques(techniques_csv: Path) -> None:
    """Ensure every technique id referenced by a rule exists in the MITRE CSV.

    A missing id is a configuration bug: the rule would create dangling edges
    in the graph artifact, breaking the HGT loader. Raises ``ValueError``.
    """
    df = pd.read_csv(techniques_csv)
    if "technique_id" not in df.columns:
        raise ValueError(
            "MITRE techniques CSV must contain a 'technique_id' column"
        )
    known = set(df["technique_id"].astype(str).tolist())
    missing = sorted(ALL_REFERENCED_TECHNIQUES - known)
    if missing:
        raise ValueError(
            "Signature rules reference unknown MITRE techniques: "
            + ", ".join(missing)
        )
