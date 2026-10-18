from pathlib import Path

import pytest

from graphslm_ids.offline.federated.attacks import build_attack
from graphslm_ids.offline.federated.config import apply_overrides, load_fl_config, unknown_keys, yaml_load
from graphslm_ids.offline.federated.defense import build_defense
from graphslm_ids.offline.federated.matrix import add_attack_impact, expand_grid, pivot_markdown
from graphslm_ids.offline.federated.simulation import VALID_MODES, _check_config, select_malicious

ROOT = Path(__file__).resolve().parents[2]
CFG_DIR = ROOT / "configs" / "fl"
GRID_SIZES = {
    "matrix_pilot.yaml": 8,
    "matrix_ch1.yaml": 27,
    "matrix_ch1_temporal.yaml": 9,
    "matrix_p0_defense.yaml": 84,
    "matrix_p0_ablation.yaml": 12,
    "matrix_p1_ratio.yaml": 18,
    "matrix_p1_group.yaml": 6,
    "matrix_p1_ch2.yaml": 96,
    "matrix_p1_seeds.yaml": 27,
    "matrix_p2_eacs.yaml": 4,
    "matrix_p2_threat.yaml": 8,
}


def test_base_config_follows_the_spec():
    cfg = load_fl_config(CFG_DIR / "fl_base.yaml")
    assert unknown_keys(cfg) == []
    assert cfg["partition"]["num_clients"] == 20 and cfg["federated"]["rounds"] == 20
    assert cfg["partition"]["scheme"] == "explicit" and cfg["client"]["eacs"]["enabled"] is False
    assert cfg["defense"]["assumed_malicious_ratio"] == 0.2


@pytest.mark.parametrize("grid_name", sorted(GRID_SIZES))
def test_every_matrix_cell_builds(grid_name):
    grid = yaml_load((CFG_DIR / grid_name).read_text(encoding="utf-8"))
    base = load_fl_config(ROOT / grid["base_config"])
    cells = expand_grid(grid)
    assert len(cells) == GRID_SIZES[grid_name]
    for cell in cells:
        cfg = apply_overrides(base, {**(grid.get("overrides") or {}), **cell["overrides"]})
        assert unknown_keys(cfg) == [], cell["name"]
        assert cfg["federated"]["mode"] in VALID_MODES and cfg["model"]["name"] in {"hgt", "mlp", "cnn"}
        assert cfg["defense"]["assumed_malicious_ratio"] == 0.2
        build_attack(cfg["attack"])
        build_defense(cfg["defense"]["aggregator"], assumed_malicious_ratio=0.2)
        select_malicious(20, cfg["attack"]["malicious_ratio"], 42, placement=cfg["attack"]["placement"], groups=cfg["partition"]["groups"])
        _check_config(cfg, None)  # e.g. no cell overrides the assumed malicious ratio


def _cells(grid_name):
    grid = yaml_load((CFG_DIR / grid_name).read_text(encoding="utf-8"))
    base = load_fl_config(ROOT / grid["base_config"])
    return [(cell, apply_overrides(base, {**(grid.get("overrides") or {}), **cell["overrides"]})) for cell in expand_grid(grid)]


def test_p0_fedavg_cells_save_their_deltas_for_offline_replay():
    for cell, cfg in _cells("matrix_p0_defense.yaml"):
        assert cfg["output"]["save_deltas"] is (cell["labels"]["defense"] == "fedavg"), cell["name"]
        assert cfg["output"]["save_deltas_dtype"] == "float16"


def test_seed_grid_has_the_seed_42_counterpart():
    seeds = {(cell["labels"]["seed"], cfg["federated"]["seed"], cfg["partition"]["seed"]) for cell, cfg in _cells("matrix_p1_seeds.yaml")}
    assert seeds == {("42", 42, 42), ("43", 43, 43), ("44", 44, 44)}


def test_all_grids_summarized_together_fill_impact_and_keep_one_run_per_pivot_cell():
    rows = []
    for grid_name in sorted(GRID_SIZES):
        for cell, cfg in _cells(grid_name):
            rows.append(
                {
                    "name": f"{grid_name}/{cell['name']}",
                    "mode": cfg["federated"]["mode"],
                    "attack": cfg["attack"]["name"],
                    "final_test_macro_f1": 0.5,
                    "final_test_macro_f1_clean": 0.5,
                    **{f"label_{k}": v for k, v in cell["labels"].items()},
                }
            )
    add_attack_impact(rows)
    for grid_name in ("matrix_p1_ratio.yaml", "matrix_p1_group.yaml", "matrix_p2_threat.yaml"):
        cells = [r for r in rows if r["name"].startswith(grid_name)]
        assert cells and all(r["label_partition"] == "p80" for r in cells)
        assert all(r["impact_final_test_macro_f1"] == 0.0 for r in cells), grid_name
    pivot_markdown(rows)  # raises if two runs would share a cell
