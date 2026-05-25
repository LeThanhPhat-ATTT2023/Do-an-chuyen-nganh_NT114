"""Unit tests for the MITRE procedure-pattern matcher.

The matcher is a thin layer over a deterministic STIX file + Aho-Corasick (or
pyahocorasick-free fallback). Contracts under test:

* Pattern extraction picks up both ``<code>`` blocks and backtick literals,
  filters by length, and lowercases.
* ``match()`` returns ``{technique_id: [patterns]}`` on positive hits and an
  empty dict on empty / unmatched payloads.
* ``weight_per_technique()`` collapses pattern hits to the calibrated
  ``min(1.0, 0.4 * n_distinct)`` curve.
"""
from __future__ import annotations

from pathlib import Path

from graphslm_ids.offline.preprocessing.procedure_matcher import (
    ProcedureMatcher,
    _extract_patterns,
    extract_technique_id_from_stix_obj,
)


def test_extract_patterns_picks_up_code_blocks_and_backticks() -> None:
    desc = (
        "Adversaries inject <code>UNION SELECT</code> into web apps. "
        "They may also run `cat /etc/passwd` directly."
    )
    pats = _extract_patterns(desc)
    assert "union select" in pats
    assert "cat /etc/passwd" in pats
    # Verify lowercasing is applied.
    assert all(p == p.lower() for p in pats)


def test_extract_patterns_filters_by_length() -> None:
    # 1-char pattern below min, 100-char pattern above max — both must be dropped.
    desc = "Try <code>a</code> or <code>" + ("z" * 100) + "</code> as edge cases."
    pats = _extract_patterns(desc)
    assert "a" not in pats
    assert ("z" * 100) not in pats


def test_extract_technique_id_from_stix_obj_finds_mitre_attack_ref() -> None:
    obj = {
        "external_references": [
            {"source_name": "other", "external_id": "OTHER-1"},
            {"source_name": "mitre-attack", "external_id": "T1190"},
        ]
    }
    assert extract_technique_id_from_stix_obj(obj) == "T1190"


def test_extract_technique_id_returns_none_when_no_mitre_ref() -> None:
    obj = {"external_references": [{"source_name": "other", "external_id": "X"}]}
    assert extract_technique_id_from_stix_obj(obj) is None
    assert extract_technique_id_from_stix_obj({}) is None


def test_match_positive_on_known_pattern(mitre_dir_fixture: Path) -> None:
    matcher = ProcedureMatcher(mitre_dir_fixture / "enterprise-attack.json")
    payload = b"GET /?q=1 UNION SELECT * FROM information_schema.tables"
    hits = matcher.match(payload)
    # Both T1190-associated patterns (union select + information_schema) fire.
    assert "T1190" in hits
    matched = set(hits["T1190"])
    assert "union select" in matched
    assert "information_schema" in matched


def test_match_empty_payload_returns_empty(mitre_dir_fixture: Path) -> None:
    matcher = ProcedureMatcher(mitre_dir_fixture / "enterprise-attack.json")
    assert matcher.match(b"") == {}


def test_match_unrelated_payload_returns_empty(mitre_dir_fixture: Path) -> None:
    matcher = ProcedureMatcher(mitre_dir_fixture / "enterprise-attack.json")
    # No fixture pattern collides with this benign text.
    assert matcher.match(b"the quick brown fox jumps over the lazy dog") == {}


def test_weight_per_technique_caps_at_one(mitre_dir_fixture: Path) -> None:
    matcher = ProcedureMatcher(mitre_dir_fixture / "enterprise-attack.json")
    # Payload that hits BOTH T1190 patterns -> n_distinct = 2 -> weight = 0.8.
    payload = b"UNION SELECT * FROM information_schema.columns"
    weights = matcher.weight_per_technique(payload)
    assert "T1190" in weights
    assert 0.0 < weights["T1190"] <= 1.0
    # Specifically the calibration curve: two distinct patterns -> 0.4 * 2 = 0.8.
    assert abs(weights["T1190"] - 0.8) < 1e-6


def test_weight_per_technique_saturates(mitre_dir_fixture: Path) -> None:
    matcher = ProcedureMatcher(mitre_dir_fixture / "enterprise-attack.json")
    # Single pattern hit -> 0.4 weight (saturation curve verified).
    payload = b"running cmd.exe /c whoami on the box"
    weights = matcher.weight_per_technique(payload)
    assert "T1059" in weights
    assert abs(weights["T1059"] - 0.4) < 1e-6


def test_match_handles_no_pattern_branch(mitre_dir_fixture: Path) -> None:
    """The fixture's T1046 entry has no <code>/backtick patterns — must never fire."""
    matcher = ProcedureMatcher(mitre_dir_fixture / "enterprise-attack.json")
    payload = b"Adversaries scan network services frequently"
    hits = matcher.match(payload)
    assert "T1046" not in hits
