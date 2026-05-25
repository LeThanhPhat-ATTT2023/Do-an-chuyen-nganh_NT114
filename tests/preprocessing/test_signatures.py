"""Unit tests for the v2 signature module (payload + flow signatures)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from graphslm_ids.offline.preprocessing.signatures import (
    match_flow_signatures,
    match_payload_signatures,
    validate_techniques,
)


def _techs(hits) -> set[str]:
    return {t for t, _ in hits}


def test_sqli_payload_triggers_t1190() -> None:
    assert "T1190" in _techs(
        match_payload_signatures(b"GET /?id=1' OR 1=1-- HTTP/1.1")
    )


def test_url_encoded_sqli_decoded_first() -> None:
    # The rule's transformations include url_decode, so encoded payloads still
    # match.
    assert "T1190" in _techs(
        match_payload_signatures(b"GET /?id=1%27%20OR%201%3D1--%20HTTP/1.1")
    )


def test_xss_script_tag_triggers_t1059_007() -> None:
    assert "T1059.007" in _techs(
        match_payload_signatures(b'<script>alert("xss")</script>')
    )


def test_shell_injection_triggers_t1059_004() -> None:
    assert "T1059.004" in _techs(
        match_payload_signatures(b"GET /?cmd=cat /etc/passwd")
    )


def test_webshell_token_triggers_t1505_003() -> None:
    assert "T1505.003" in _techs(
        match_payload_signatures(b"POST /upload eval(base64_decode('...'));")
    )


def test_empty_payload_returns_no_hits() -> None:
    assert match_payload_signatures(b"") == []


def test_synscan_flow_triggers_t1046() -> None:
    flow = pd.Series(
        {
            "proto": 0,
            "fan_dst_port": 1200,
            "fan_dst_ip": 1,
            "r_syn": 0.95,
            "plen_pay_mean": 0,
            "pkts_per_s": 5,
            "r_rst": 0.0,
            "r_psh": 0.0,
            "tot_pkts": 1200,
            "dur": 60.0,
            "fwd_bytes": 60_000,
            "bwd_bytes": 0,
        }
    )
    assert "T1046" in _techs(match_flow_signatures(flow))


def test_icmp_pingsweep_flow_triggers_t1018() -> None:
    flow = pd.Series(
        {
            "proto": 2,
            "fan_dst_ip": 80,
            "fan_dst_port": 1,
            "r_syn": 0,
            "plen_pay_mean": 8,
            "pkts_per_s": 50,
            "r_rst": 0,
            "r_psh": 0,
            "tot_pkts": 100,
            "dur": 2.0,
            "fwd_bytes": 800,
            "bwd_bytes": 0,
        }
    )
    assert "T1018" in _techs(match_flow_signatures(flow))


def test_benign_flow_does_not_trigger_any_flow_signature() -> None:
    flow = pd.Series(
        {
            "proto": 0,
            "fan_dst_port": 1,
            "fan_dst_ip": 1,
            "r_syn": 0.1,
            "plen_pay_mean": 200,
            "pkts_per_s": 10,
            "r_rst": 0.0,
            "r_psh": 0.4,
            "tot_pkts": 30,
            "dur": 3.0,
            "fwd_bytes": 6_000,
            "bwd_bytes": 6_000,
        }
    )
    assert match_flow_signatures(flow) == []


def test_validate_techniques_against_real_mitre_csv() -> None:
    csv = Path("data/mitre/mitre_techniques.csv")
    if not csv.exists():
        pytest.skip("data/mitre/mitre_techniques.csv not present locally")
    validate_techniques(csv)  # must not raise
