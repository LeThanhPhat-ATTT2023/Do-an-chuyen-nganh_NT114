"""MITRE ATT&CK procedure-pattern matcher (Source 2 of the v3 PMI Ensemble).

Mines literal substring patterns out of the STIX ``enterprise-attack.json``
file shipped under ``data/mitre/`` — specifically the ``<code>...</code>``
blocks and backtick-delimited inline literals embedded in each attack-pattern
description. These are the *procedures* (in ATT&CK parlance the "P" of the
TTP triad) that adversaries are observed running in the wild, and they tend
to be high-fidelity needles for payload matching (e.g. ``information_schema``,
``/etc/passwd``, ``UNION SELECT``).

Pipeline:

1. ``__init__`` parses the STIX JSON, extracts patterns, filters to
   ``3 <= len <= 80`` (lowercase), and builds an Aho-Corasick automaton via
   ``pyahocorasick`` if available.
2. ``match()`` lowercases the payload (latin-1, errors ignored) and returns
   ``{technique_id: [patterns hit]}``.
3. ``weight_per_technique()`` collapses that to a [0, 1] weight via
   ``min(1.0, 0.4 * n_distinct_patterns)``.

Fallback when ``pyahocorasick`` is unavailable: a Python-level multi-substring
search using ``str.find`` against the lowercase payload. This is O(P * L) per
packet where P = #patterns (~thousands) and L = payload length. On the
600K-packet graph build that adds roughly 5-8 minutes vs the ~30 seconds
the C-accelerated automaton would take. Acceptable for the once-off
preprocessing run; not viable at runtime inference.

Memory budget: ~1 GB peak when the automaton is built (pyahocorasick keeps an
internal trie + failure links per pattern). Without pyahocorasick the
fallback list stays in ~30 MB.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

_LOG = logging.getLogger(__name__)

# Extract <code>...</code> blocks and backtick-delimited inline literals. The
# STIX descriptions use both forms interchangeably for procedures. We do not
# use HTML parsing here because the descriptions are Markdown-ish, not real
# HTML, and the regex is both cheaper and immune to malformed nesting.
_CODE_BLOCK_RE = re.compile(r"<code>(.*?)</code>", flags=re.IGNORECASE | re.DOTALL)
_BACKTICK_RE = re.compile(r"`([^`\n]+?)`")

_MIN_PATTERN_LEN = 3
_MAX_PATTERN_LEN = 80


def extract_technique_id_from_stix_obj(obj: dict) -> str | None:
    """Return the ATT&CK technique ID (e.g. ``T1190``) for a STIX object.

    STIX models the relation as a list of external references; the canonical
    MITRE ID lives on the reference whose ``source_name == "mitre-attack"``.
    Returns ``None`` if no such reference exists (sub-techniques sometimes
    appear in revoked / deprecated state and lack a stable ID).
    """
    for ref in obj.get("external_references", []) or []:
        if ref.get("source_name") == "mitre-attack":
            ext_id = ref.get("external_id")
            if ext_id:
                return str(ext_id)
    return None


def _extract_patterns(description: str) -> list[str]:
    """Pull <code> blocks + backtick literals; filter to valid pattern length."""
    if not description:
        return []
    found: list[str] = []
    for m in _CODE_BLOCK_RE.finditer(description):
        found.append(m.group(1))
    for m in _BACKTICK_RE.finditer(description):
        found.append(m.group(1))
    out: list[str] = []
    for p in found:
        # Whitespace-collapse so multi-line code blocks become a single
        # searchable string; matchers operate on raw bytes that don't contain
        # arbitrary newlines anyway.
        p = " ".join(p.split()).lower()
        if _MIN_PATTERN_LEN <= len(p) <= _MAX_PATTERN_LEN:
            out.append(p)
    return out


class ProcedureMatcher:
    """Aho-Corasick (or fallback) matcher over MITRE procedure literals.

    Use :func:`match` to get raw pattern hits per technique, or
    :func:`weight_per_technique` for the calibrated [0, 1] evidence weight
    the v3 ensemble expects.
    """

    def __init__(self, stix_path: Path) -> None:
        """Parse STIX, mine patterns, build matcher.

        Args:
            stix_path: path to ``enterprise-attack.json`` (MITRE-published).
        """
        stix_path = Path(stix_path)
        with stix_path.open("r", encoding="utf-8") as f:
            stix = json.load(f)

        # Map: pattern -> set of technique_ids that mention that pattern.
        # Using a set protects against a pattern appearing across multiple
        # techniques (e.g. ``cmd.exe`` shows up in many) — we want all of
        # them to fire on a payload match, not just the first one parsed.
        pattern_to_techs: dict[str, set[str]] = {}
        for obj in stix.get("objects", []):
            if obj.get("type") != "attack-pattern":
                continue
            tech_id = extract_technique_id_from_stix_obj(obj)
            if not tech_id:
                continue
            for pattern in _extract_patterns(obj.get("description", "")):
                pattern_to_techs.setdefault(pattern, set()).add(tech_id)

        self._pattern_to_techs: dict[str, set[str]] = pattern_to_techs
        self._patterns: list[str] = list(pattern_to_techs.keys())
        _LOG.info(
            "ProcedureMatcher: extracted %d patterns covering %d techniques",
            len(self._patterns),
            len({t for techs in pattern_to_techs.values() for t in techs}),
        )

        # Try the fast path first; fall back gracefully on ImportError so
        # this module is importable in environments where pyahocorasick
        # could not be installed (the project's pinned dependency list does
        # not include it).
        self._automaton = None
        try:
            import ahocorasick  # type: ignore

            automaton = ahocorasick.Automaton()
            for pat in self._patterns:
                automaton.add_word(pat, pat)
            automaton.make_automaton()
            self._automaton = automaton
            _LOG.info("ProcedureMatcher: using pyahocorasick automaton")
        except ImportError:
            _LOG.warning(
                "ProcedureMatcher: pyahocorasick not installed, falling back "
                "to O(P*L) Python multi-substring search (~5-8 min for 600K "
                "packets vs ~30 sec for the automaton)"
            )

    def match(self, payload: bytes) -> dict[str, list[str]]:
        """Return ``{technique_id: [matched_patterns]}`` for ``payload``.

        Payload is lowercased via latin-1 decode (``errors='ignore'``) before
        matching. Returns an empty dict on empty / no-hit payloads.
        """
        if not payload:
            return {}
        text = payload.decode("latin-1", errors="ignore").lower()

        hits_by_pattern: set[str] = set()
        if self._automaton is not None:
            # pyahocorasick yields ``(end_index, value)`` tuples where value
            # is whatever we stored — we stored the pattern string itself.
            for _end_idx, pat in self._automaton.iter(text):
                hits_by_pattern.add(pat)
        else:
            # Fallback: linear scan. Use ``str.find`` which is C-implemented
            # and significantly faster than ``re.search`` for literal needles.
            for pat in self._patterns:
                if text.find(pat) != -1:
                    hits_by_pattern.add(pat)

        out: dict[str, list[str]] = {}
        for pat in hits_by_pattern:
            for tech_id in self._pattern_to_techs[pat]:
                out.setdefault(tech_id, []).append(pat)
        return out

    def weight_per_technique(self, payload: bytes) -> dict[str, float]:
        """Collapse :func:`match` to a calibrated [0, 1] weight per technique.

        Weight is ``min(1.0, 0.4 * len(set(patterns)))`` — saturates at 3
        distinct patterns. The 0.4 multiplier was chosen so a single
        high-confidence hit clears the v3 ensemble's ``tau_edge = 0.4``
        threshold only when corroborated by a PMI vote.
        """
        hits = self.match(payload)
        return {
            tech_id: min(1.0, 0.4 * len(set(patterns)))
            for tech_id, patterns in hits.items()
        }
