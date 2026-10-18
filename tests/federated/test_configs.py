from pathlib import Path

import pytest

from graphslm_ids.offline.federated.attacks import build_attack
from graphslm_ids.offline.federated.config import apply_overrides, load_fl_config, unknown_keys, yaml_load
from graphslm_ids.offline.federated.defense import build_defense
from graphslm_ids.offline.federated.matrix import expand_grid
from graphslm_ids.offline.federated.simulation import VALID_MODES, select_malicious

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
    "matrix_p1_seeds.yaml": 18,
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
