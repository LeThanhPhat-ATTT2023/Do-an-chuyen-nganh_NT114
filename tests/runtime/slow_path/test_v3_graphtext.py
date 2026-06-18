import pandas as pd

from graphslm_ids.offline.preprocessing.ensemble import (
    build_pmi_lookup_from_table,
    lookup_pmi_per_packet,
    lookup_pmi_per_packet_with_tokens,
)


def _lookup():
    df = pd.DataFrame([
        {"token": "t:select", "technique": "T1190", "family": "injection", "weight": 0.9},
    ])
    return build_pmi_lookup_from_table(df)


def test_pmi_with_tokens_matches_base_weights():
    lk = _lookup()
    payload = b"... select ..."
    base = lookup_pmi_per_packet(payload, lk)
    prov = lookup_pmi_per_packet_with_tokens(payload, lk)
    # same techniques + same (family, weight)
    assert set(prov) == set(base)
    for tech, (family, weight, tokens) in prov.items():
        assert (family, weight) == base[tech]
        assert "t:select" in tokens


def test_pmi_with_tokens_empty_payload():
    assert lookup_pmi_per_packet_with_tokens(b"", _lookup()) == {}
