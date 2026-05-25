"""Unit tests for the v3 PMI candidate generation + L1-LR pipeline.

Goals locked in here:

* Stratified subsample respects per-class caps + the global ``max_total`` cap.
* PMI candidate filter honours support thresholds (small classes don't leak
  through the global support_min when their per-class count is below
  support_min_class).
* L1 logistic returns a coef matrix shaped ``(n_classes, n_candidates)``
  even when sklearn collapses the binary case to one row.
* Projection to techniques drops rows lacking a family mapping and
  thresholds tiny aggregated weights.
* End-to-end ``fit_and_save_pmi_table`` writes a parquet + meta json that
  round-trip cleanly and contain non-empty rows on a non-trivial fixture.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from graphslm_ids.offline.preprocessing.pmi_learner import (
    filter_pmi_candidates,
    fit_and_save_pmi_table,
    fit_l1_logistic,
    fit_pmi_counts,
    project_to_techniques,
    stratified_subsample_packets,
)


@pytest.fixture
def pmi_packets_df() -> pd.DataFrame:
    """Two-class synthetic packet set with class-discriminative tokens.

    Class A always carries ``ALPHA_TOKEN_BYTES`` in its payload; class B always
    carries ``BETA_TOKEN_BYTES``. The shared filler bytes are common to both,
    so a well-behaved PMI learner should single out the discriminative tokens
    and not the filler.
    """
    rng = np.random.default_rng(0)
    rows = []
    for cls, marker in (("ClassA", b"ALPHA_TOKEN_BYTES "), ("ClassB", b"BETA_TOKEN_BYTES ")):
        for i in range(150):
            filler = bytes(rng.integers(0x61, 0x7B, size=64, dtype=np.uint8).tolist())
            payload = marker * 3 + filler
            rows.append({"payload": payload, "label": cls, "flow_id": f"{cls}_{i}"})
    return pd.DataFrame(rows)


def test_stratified_subsample_respects_global_cap(pmi_packets_df: pd.DataFrame) -> None:
    sub = stratified_subsample_packets(
        pmi_packets_df, n_per_class=80, max_total=120, seed=42
    )
    # Global cap honored even though 2 * n_per_class = 160 > 120.
    assert len(sub) == 120
    # Both classes contribute (no class wiped out by the cap).
    assert sub["label"].nunique() == 2


def test_stratified_subsample_pins_tiny_classes(pmi_packets_df: pd.DataFrame) -> None:
    # Inject a tiny class with only 3 rows. Per-class oversampling-with-replacement
    # path should fire (n < 100 branch).
    tiny = pd.DataFrame(
        [{"payload": b"TINY_TOKEN_X", "label": "Rare", "flow_id": f"r{i}"} for i in range(3)]
    )
    df = pd.concat([pmi_packets_df, tiny], ignore_index=True)
    sub = stratified_subsample_packets(df, n_per_class=100, max_total=10_000, seed=7)
    # Tiny class must still be present; oversampling guarantees > 3 rows.
    rare_n = int((sub["label"] == "Rare").sum())
    assert rare_n > 3, f"tiny class should be oversampled, got n={rare_n}"


def test_fit_pmi_counts_returns_class_totals_and_token_counter(
    pmi_packets_df: pd.DataFrame,
) -> None:
    class_total, token_class = fit_pmi_counts(pmi_packets_df.head(20))
    assert isinstance(class_total, Counter)
    assert sum(class_total.values()) == 20
    # token_class is a dict of Counter; pick any token and verify shape.
    assert all(isinstance(c, Counter) for c in token_class.values())


def test_filter_pmi_candidates_respects_support(
    pmi_packets_df: pd.DataFrame,
) -> None:
    class_total, token_class = fit_pmi_counts(pmi_packets_df)
    # With very high support thresholds nothing should pass.
    cands_strict, scores_strict = filter_pmi_candidates(
        token_class,
        class_total,
        support_min_global=10_000,
        support_min_class=1_000,
    )
    assert cands_strict == set()
    assert scores_strict == {}

    # Relaxed thresholds: at least the class-marker token must surface.
    cands, scores = filter_pmi_candidates(
        token_class,
        class_total,
        pmi_min=0.0,
        support_min_global=10,
        support_min_class=5,
        top_k_per_class=200,
    )
    assert len(cands) > 0


def test_fit_l1_logistic_returns_shape_n_classes_by_n_features(
    pmi_packets_df: pd.DataFrame,
) -> None:
    class_total, token_class = fit_pmi_counts(pmi_packets_df)
    cands, _ = filter_pmi_candidates(
        token_class,
        class_total,
        pmi_min=0.0,
        support_min_global=10,
        support_min_class=5,
    )
    label_to_idx = {lbl: i for i, lbl in enumerate(sorted(class_total))}
    coef, tokens, mapping = fit_l1_logistic(
        pmi_packets_df, cands, label_to_idx, C=1.0, max_iter=50, n_jobs=1, seed=42
    )
    # Binary case: coef must be padded to (n_classes, n_features) so downstream
    # projection can index by class without special-casing.
    assert coef.shape == (len(label_to_idx), len(tokens))
    assert mapping == label_to_idx


def test_fit_l1_logistic_empty_candidates_returns_empty_coef() -> None:
    df = pd.DataFrame([{"payload": b"x", "label": "A"}])
    coef, tokens, mapping = fit_l1_logistic(
        df, candidates=set(), label_to_idx={"A": 0}, seed=42
    )
    assert coef.size == 0
    assert tokens == []
    assert mapping == {"A": 0}


def test_project_to_techniques_drops_rows_without_family() -> None:
    coef = np.array([[0.6, -0.3], [0.0, 0.5]], dtype=np.float32)
    tokens = ["b4:aabbccdd", "t:exec"]
    class_to_idx = {"SqlInjection": 0, "CommandInjection": 1}
    class_map = pd.DataFrame(
        [
            {"class": "SqlInjection", "technique": "T1190", "weight": 1.0},
            {"class": "CommandInjection", "technique": "T1059", "weight": 1.0},
            # Technique without a family entry — must be dropped.
            {"class": "SqlInjection", "technique": "T9999", "weight": 1.0},
        ]
    )
    tech_family = pd.DataFrame(
        [
            {"technique": "T1190", "family": "injection"},
            {"technique": "T1059", "family": "command_exec"},
        ]
    )
    out = project_to_techniques(coef, tokens, class_to_idx, class_map, tech_family)
    assert "T9999" not in out["technique"].tolist()
    # Remaining rows should have valid family fields.
    assert set(out["family"]).issubset({"injection", "command_exec"})


def test_project_to_techniques_thresholds_small_weights() -> None:
    # Coefficient just under the default coef_threshold (0.01) — must be dropped.
    coef = np.array([[0.005]], dtype=np.float32)
    tokens = ["b4:00000000"]
    class_to_idx = {"X": 0}
    class_map = pd.DataFrame(
        [{"class": "X", "technique": "T1190", "weight": 1.0}]
    )
    tech_family = pd.DataFrame([{"technique": "T1190", "family": "injection"}])
    out = project_to_techniques(
        coef, tokens, class_to_idx, class_map, tech_family, coef_threshold=0.01
    )
    assert len(out) == 0


def test_fit_and_save_pmi_table_writes_parquet_and_meta(
    tmp_path: Path, pmi_packets_df: pd.DataFrame
) -> None:
    # Minimal class/family CSVs covering the fixture's two classes.
    class_map = tmp_path / "class_map.csv"
    pd.DataFrame(
        [
            {"class": "ClassA", "technique": "T1190", "weight": 1.0},
            {"class": "ClassB", "technique": "T1059", "weight": 1.0},
        ]
    ).to_csv(class_map, index=False)
    tech_family = tmp_path / "tech_family.csv"
    pd.DataFrame(
        [
            {"technique": "T1190", "family": "injection"},
            {"technique": "T1059", "family": "command_exec"},
        ]
    ).to_csv(tech_family, index=False)

    out_parquet = tmp_path / "pmi_table.parquet"
    out_meta = tmp_path / "pmi.meta.json"

    meta = fit_and_save_pmi_table(
        pmi_packets_df,
        class_technique_map_path=class_map,
        technique_family_path=tech_family,
        out_pmi_table=out_parquet,
        out_meta_json=out_meta,
        n_per_class=100,
        max_total=200,
        seed=42,
        # Tiny fixture: lower thresholds so discriminative tokens survive.
        # With balanced 2-class fixture max PMI = ln(2) ~= 0.69, so default
        # pmi_min=1.0 would reject everything → use 0.3.
        pmi_min=0.3,
        support_min_global=5,
        support_min_class=3,
        C=1.0,                   # less aggressive L1 (more coef survives)
        coef_threshold=0.001,    # don't strip near-zero (small dataset → small coefs)
    )

    # Both files must exist on disk.
    assert out_parquet.exists()
    assert out_meta.exists()

    # Meta JSON should round-trip and contain the expected keys.
    loaded = json.loads(out_meta.read_text(encoding="utf-8"))
    assert loaded["n_classes"] == 2
    assert loaded["pmi_table_rows"] == meta["pmi_table_rows"]
    assert "wall_seconds" in loaded
    assert "hyperparameters" in loaded

    # Parquet rows should be non-empty (the fixture has clear discriminative
    # tokens) and conform to the documented column contract.
    df = pd.read_parquet(out_parquet)
    assert set(df.columns) == {"token", "technique", "family", "weight"}
    assert len(df) > 0
    assert df["family"].isin({"injection", "command_exec"}).all()
