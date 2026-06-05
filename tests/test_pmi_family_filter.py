"""Unit tests for ``apply_family_specificity_filter`` in pmi_learner.

The filter drops cross-family-smeared PMI token rows: for each token it sums
``|weight|`` per family and keeps a row iff its family's summed |weight| is
>= ``tau * max_family_weight`` for that token.
"""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from graphslm_ids.offline.preprocessing.pmi_learner import (
    apply_family_specificity_filter,
)

COLS = ["token", "technique", "family", "weight"]


def _df(rows):
    return pd.DataFrame(rows, columns=COLS)


def test_tau_zero_is_noop():
    """tau == 0.0 returns the input unchanged (same rows)."""
    df = _df(
        [
            ("tok", "T1", "A", 1.0),
            ("tok", "T2", "B", -0.5),
        ]
    )
    out = apply_family_specificity_filter(df, tau=0.0)
    assert_frame_equal(out, df)


def test_negative_tau_is_noop():
    """tau < 0.0 also disables the filter."""
    df = _df([("tok", "T1", "A", 1.0), ("tok", "T2", "B", 0.5)])
    out = apply_family_specificity_filter(df, tau=-1.0)
    assert_frame_equal(out, df)


def test_empty_dataframe_returns_empty():
    """An empty DataFrame with the correct columns is returned empty."""
    df = _df([])
    out = apply_family_specificity_filter(df, tau=0.6)
    assert out.empty
    assert list(out.columns) == COLS


def test_promiscuous_token_dropped_specific_token_kept():
    """A promiscuous token keeps only its dominant family; a family-specific
    token (single family) is always kept."""
    df = _df(
        [
            # promiscuous token "smear": A total=1.0, B=0.1, C=0.05
            ("smear", "T_A", "A", 1.0),
            ("smear", "T_B", "B", -0.1),
            ("smear", "T_C", "C", 0.05),
            # family-specific token "spec": single family D
            ("spec", "T_D", "D", 0.3),
        ]
    )
    out = apply_family_specificity_filter(df, tau=0.6)

    # tau*fmax = 0.6*1.0 = 0.6 -> only family A (1.0) survives for "smear".
    smear = out[out["token"] == "smear"]
    assert set(smear["family"]) == {"A"}
    assert len(smear) == 1

    # family-specific token: fmax==its own weight, always >= tau*fmax -> kept.
    spec = out[out["token"] == "spec"]
    assert len(spec) == 1
    assert spec.iloc[0]["family"] == "D"


def test_threshold_boundary_is_inclusive():
    """A secondary family whose summed |weight| equals EXACTLY tau*fmax is kept
    (the comparison is >=, inclusive). Uses fp-exact values."""
    # fmax = 1.0, tau = 0.5 -> threshold = 0.5; secondary family B = 0.5 exactly.
    df = _df(
        [
            ("tok", "T_A", "A", 1.0),
            ("tok", "T_B", "B", -0.5),  # abs == 0.5 == 0.5*1.0
            ("tok", "T_C", "C", 0.4),   # below threshold -> dropped
        ]
    )
    out = apply_family_specificity_filter(df, tau=0.5)
    fams = set(out[out["token"] == "tok"]["family"])
    assert fams == {"A", "B"}  # B is inclusive-kept, C dropped


def test_aggregation_sums_within_family():
    """A family that dominates only via several small rows summing above the
    threshold is kept -> verifies per-family SUM, not MAX."""
    df = _df(
        [
            # family A: three small rows summing to 0.9 (max single row 0.3)
            ("tok", "T_A1", "A", 0.3),
            ("tok", "T_A2", "A", -0.3),
            ("tok", "T_A3", "A", 0.3),
            # family B: one row of 0.5 (larger single row than any A row)
            ("tok", "T_B", "B", 0.5),
        ]
    )
    # fam_weight: A=0.9, B=0.5 -> fmax=0.9, tau*fmax=0.7*0.9=0.63.
    # B=0.5 < 0.63 -> dropped. If it MAXed per family (A=0.3) B would survive.
    out = apply_family_specificity_filter(df, tau=0.7)
    kept = out[out["token"] == "tok"]
    assert set(kept["family"]) == {"A"}
    # all three A rows survive
    assert len(kept) == 3


def test_missing_column_raises_valueerror():
    """Dropping any required column raises ValueError (when tau > 0)."""
    for drop in COLS:
        df = _df([("tok", "T1", "A", 1.0)]).drop(columns=[drop])
        with pytest.raises(ValueError):
            apply_family_specificity_filter(df, tau=0.6)


def test_deterministic():
    """Two identical calls yield identical results."""
    df = _df(
        [
            ("smear", "T_A", "A", 1.0),
            ("smear", "T_B", "B", -0.1),
            ("smear", "T_C", "C", 0.05),
            ("spec", "T_D", "D", 0.3),
            ("tok2", "T_E", "E", 0.8),
            ("tok2", "T_F", "F", 0.79),
        ]
    )
    out1 = apply_family_specificity_filter(df, tau=0.6)
    out2 = apply_family_specificity_filter(df, tau=0.6)
    assert_frame_equal(out1, out2)
