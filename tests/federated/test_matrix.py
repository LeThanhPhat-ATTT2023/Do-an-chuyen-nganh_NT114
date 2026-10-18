import importlib.util
import itertools
from pathlib import Path

import pytest
import yaml

from graphslm_ids.offline.federated import matrix
from graphslm_ids.offline.federated.matrix import (
    add_attack_impact,
    expand_grid,
    pivot_markdown,
    run_grid,
    summarize,
    summary_row,
    write_csv,
)

ROOT = Path(__file__).resolve().parents[2]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "fl" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _grid(smoke_cfg, tmp_path):
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(smoke_cfg("unused", **{"federated.rounds": 1})), encoding="utf-8")
    grid = {
        "base_config": str(base_path),
        "output_root": str(tmp_path / "matrix"),
        "name_template": "{attack}__m{malicious_ratio}__{defense}",
        "overrides": {"output.save_checkpoints": False},
        "axes": {
            "attack": {"none": {"attack.name": "none"}, "sign_flip": {"attack.name": "sign_flip"}},
            "malicious_ratio": {"0.34": {"attack.malicious_ratio": 0.34}},
            "defense": {"fedavg": {"defense.aggregator": {"name": "fedavg"}}},
        },
    }
    grid_path = tmp_path / "grid.yaml"
    grid_path.write_text(yaml.safe_dump(grid), encoding="utf-8")
    return grid_path


def test_expand_grid_dedupes_no_attack_cells():
    grid = {
        "name_template": "{attack}__m{malicious_ratio}__{defense}",
        "axes": {
            "attack": {"none": {"attack.name": "none"}, "alie": {"attack.name": "alie"}},
            "malicious_ratio": {"0.1": {"attack.malicious_ratio": 0.1}, "0.2": {"attack.malicious_ratio": 0.2}},
            "defense": {"fedavg": {}},
        },
    }
    cells = expand_grid(grid)
    assert [c["name"] for c in cells] == ["none__m0__fedavg", "alie__m0.1__fedavg", "alie__m0.2__fedavg"]
    assert cells[0]["overrides"]["attack.malicious_ratio"] == 0.0
    assert cells[1]["labels"] == {"attack": "alie", "malicious_ratio": "0.1", "defense": "fedavg"}


def test_expand_grid_orders_by_split_model_partition_seed():
    grid = {
        "name_template": "{model}__{partition}__{seed}",
        "axes": {
            "seed": {"43": {}, "42": {}},
            "partition": {"p80": {}, "p0": {}},
            "model": {"mlp": {}, "hgt": {}},
        },
    }
    names = [c["name"] for c in expand_grid(grid)]
    assert names == [f"{m}__{p}__{s}" for m in ("hgt", "mlp") for p in ("p0", "p80") for s in ("42", "43")]


def test_run_grid_runs_resumes_and_summarizes(smoke_cfg, tmp_path):
    grid_path = _grid(smoke_cfg, tmp_path)
    assert [r["status"] for r in run_grid(grid_path, dry_run=True)] == ["planned", "planned"]
    assert [r["status"] for r in run_grid(grid_path)] == ["done", "done"]
    assert [r["status"] for r in run_grid(grid_path)] == ["skipped", "skipped"]
    assert not (tmp_path / "matrix" / "none__m0__fedavg" / "global_final.pt").exists()
    rows = {r["name"]: r for r in summarize(tmp_path / "matrix")}
    assert rows["none__m0__fedavg"]["impact_final_test_macro_f1"] == 0.0
    assert rows["sign_flip__m0.34__fedavg"]["impact_final_test_macro_f1"] is not None
    assert "screen" not in rows["none__m0__fedavg"] and rows["none__m0__fedavg"]["model"] == "hgt"
    assert "final_test_benign_far" in rows["none__m0__fedavg"]
    assert "downweight_rate_benign" in rows["none__m0__fedavg"]
    write_csv(list(rows.values()), tmp_path / "m.csv")
    assert (tmp_path / "m.csv").read_text(encoding="utf-8").startswith("name,")
    md = pivot_markdown(list(rows.values()), row_keys=("label_attack",), col_key="label_malicious_ratio", group_key="label_defense")
    assert "sign_flip" in md and "| attack |" in md


def test_matrix_scripts(smoke_cfg, tmp_path, capsys):
    grid_path = _grid(smoke_cfg, tmp_path)
    assert _load_script("run_matrix").main(["--grid", str(grid_path), "--dry-run"]) == 0
    assert "2 cells" in capsys.readouterr().out
    assert _load_script("run_matrix").main(["--grid", str(grid_path), "--only", "none__*"]) == 0
    assert _load_script("summarize_matrix").main(["--root", str(tmp_path / "matrix")]) == 0
    assert (tmp_path / "matrix" / "matrix_results.csv").exists()
    md = (tmp_path / "matrix" / "matrix_results.md").read_text(encoding="utf-8")
    assert "impact_final_test_macro_f1_clean" in md


def test_failed_cell_is_rerun_on_resume(smoke_cfg, tmp_path):
    grid_path = _grid(smoke_cfg, tmp_path)
    stale = tmp_path / "matrix" / "none__m0__fedavg"
    stale.mkdir(parents=True)
    (stale / "error.txt").write_text("boom", encoding="utf-8")
    statuses = {r["name"]: r["status"] for r in run_grid(grid_path, only="none__*")}
    assert statuses == {"none__m0__fedavg": "done"}
    assert (stale / "fl_summary.json").exists() and not (stale / "error.txt").exists()


def test_attack_impact_matches_on_every_other_label():
    rows = [
        {"name": "a", "mode": "fl", "attack": "none", "label_model": "hgt", "label_defense": "x", "final_test_macro_f1": 0.9, "final_test_macro_f1_clean": 0.8},
        {"name": "b", "mode": "fl", "attack": "alie", "label_model": "hgt", "label_defense": "x", "final_test_macro_f1": 0.6, "final_test_macro_f1_clean": 0.5},
        {"name": "c", "mode": "fl", "attack": "alie", "label_model": "mlp", "label_defense": "x", "final_test_macro_f1": 0.6, "final_test_macro_f1_clean": 0.5},
    ]
    out = {r["name"]: r for r in add_attack_impact(rows)}
    assert out["b"]["impact_final_test_macro_f1"] == pytest.approx(0.3)
    assert out["c"]["impact_final_test_macro_f1"] is None


def test_attack_impact_ignores_attack_ratio_and_placement_labels():
    rows = [
        {"name": "a", "mode": "fl", "attack": "none", "label_defense": "x", "label_attack": "none", "label_malicious_ratio": "0", "label_placement": "random", "final_test_macro_f1": 0.9},
        {"name": "b", "mode": "fl", "attack": "alie", "label_defense": "x", "label_attack": "alie", "label_malicious_ratio": "0.3", "label_placement": "g2", "final_test_macro_f1": 0.6},
    ]
    out = {r["name"]: r for r in add_attack_impact(rows)}
    assert out["b"]["impact_final_test_macro_f1"] == pytest.approx(0.3)


def test_summary_row_downweight_rate_and_local_only_spread():
    fl = {
        "experiment": "x", "mode": "fl", "model": "hgt", "aggregator": "raguard", "attack": "alie", "labels": {"seed": "42"},
        "final": {"test": {"macro_f1": 0.5, "clean": {"macro_f1": 0.4}, "family_recall": {"web": 0.1, "ddos": 0.2}}},
        "detection": {"tp": 1, "fn": 1, "fp": 1, "tn": 7, "downweighted_benign": 3, "tpr": 0.5, "fpr": 0.125},
    }
    row = summary_row(fl)
    assert row["downweight_rate_benign"] == pytest.approx(3 / 8) and row["label_seed"] == "42"
    assert row["final_test_recall_web"] == 0.1 and row["final_test_recall_ddos"] == 0.2
    assert row["benign_false_rejection"] == 0.125
    fl["detection"] = {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "downweighted_benign": 0}
    assert summary_row(fl)["downweight_rate_benign"] is None
    lo = summary_row({
        "experiment": "y", "mode": "local_only", "model": "mlp", "aggregator": None, "attack": "none", "labels": {},
        "aggregate": {"test_macro_f1": {"mean": 0.3, "std": 0.1, "min": 0.2, "max": 0.4}, "test_macro_f1_clean": {"mean": 0.25}},
    })
    assert lo["final_test_macro_f1"] == 0.3 and lo["std_test_macro_f1"] == 0.1
    assert lo["min_test_macro_f1"] == 0.2 and lo["max_test_macro_f1"] == 0.4


def test_run_grid_rejects_a_typo_before_training(smoke_cfg, tmp_path):
    grid_path = _grid(smoke_cfg, tmp_path)
    grid = yaml.safe_load(grid_path.read_text(encoding="utf-8"))
    grid["axes"]["defense"]["fedavg"] = {"defense.agregator": {"name": "fedavg"}}
    grid_path.write_text(yaml.safe_dump(grid), encoding="utf-8")
    with pytest.raises(ValueError, match="agregator"):
        run_grid(grid_path)
    assert not (tmp_path / "matrix").exists()


def _row(name, value, **labels):
    return {"name": name, "mode": "fl", "attack": "none", "final_test_macro_f1": value, **{f"label_{k}": v for k, v in labels.items()}}


def test_pivot_gives_every_run_its_own_cell():
    combos = list(itertools.product(("centralized", "local_only", "fl"), ("hgt", "mlp", "cnn"), ("p0", "p50", "p80")))
    rows = [_row(f"{m}__{mo}__{p}", 0.01 * (i + 1), mode=m, model=mo, partition=p, defense="fedavg") for i, (m, mo, p) in enumerate(combos)]
    md = pivot_markdown(rows)
    assert md.count("| mode | model | fedavg |") == 3
    for i, (m, mo, _p) in enumerate(combos):
        assert md.count(f"| {m} | {mo} | {0.01 * (i + 1):.4f} |") == 1
    # A label missing from some rows (runs of two grid roots) is a blank, not a merge.
    mixed = pivot_markdown([*rows, _row("temporal_fl_p0", 0.99, mode="fl", partition="p0", defense="fedavg")])
    assert "| fl |  | 0.9900 |" in mixed and "| fl | hgt | 0.1900 |" in mixed


def test_pivot_rejects_two_runs_in_one_cell():
    rows = [_row("a", 0.5, attack="alie", defense="fedavg", partition="p80"), _row("b", 0.6, attack="alie", defense="fedavg", partition="p80")]
    with pytest.raises(ValueError, match="'a'.*'b'|'b'.*'a'"):
        pivot_markdown(rows)


def test_attack_impact_matches_across_grid_roots():
    clean = {"final_test_macro_f1": 0.9, "final_test_macro_f1_clean": 0.8}
    rows = [
        {"name": "p0", "mode": "fl", "attack": "none", "label_partition": "p80", "label_attack": "none", "label_defense": "raguard", **clean},
        {"name": "p1_ratio", "mode": "fl", "attack": "alie", "label_partition": "p80", "label_defense": "raguard", "label_attack": "alie",
         "label_malicious_ratio": "0.3", "final_test_macro_f1": 0.5, "final_test_macro_f1_clean": 0.4},
        {"name": "p1_group", "mode": "fl", "attack": "alie", "label_partition": "p80", "label_defense": "raguard", "label_attack": "alie",
         "label_placement": "g2", "final_test_macro_f1": 0.6, "final_test_macro_f1_clean": 0.5},
        {"name": "p2_threat", "mode": "fl", "attack": "min_max", "label_partition": "p80", "label_defense": "raguard", "label_attack": "min_max",
         "label_scenario": "late_start", "final_test_macro_f1": 0.7, "final_test_macro_f1_clean": 0.6},
    ]
    out = {r["name"]: r for r in add_attack_impact(rows)}
    assert out["p1_ratio"]["impact_final_test_macro_f1"] == pytest.approx(0.4)
    assert out["p1_group"]["impact_final_test_macro_f1"] == pytest.approx(0.3)
    assert out["p2_threat"]["impact_final_test_macro_f1"] == pytest.approx(0.2)
    assert out["p2_threat"]["impact_final_test_macro_f1_clean"] == pytest.approx(0.2)


def test_failed_cell_prints_the_first_line_of_the_error(smoke_cfg, tmp_path, monkeypatch, capsys):
    grid_path = _grid(smoke_cfg, tmp_path)

    def boom(cfg, *, out_dir, context):
        raise ValueError("bad thing\nsecond line")

    monkeypatch.setattr(matrix, "run_experiment", boom)
    assert run_grid(grid_path, only="none__*") == [{"name": "none__m0__fedavg", "status": "failed"}]
    out = capsys.readouterr().out
    assert "[matrix] FAILED none__m0__fedavg: ValueError: bad thing\n" in out and "second line" not in out
    assert "second line" in (tmp_path / "matrix" / "none__m0__fedavg" / "error.txt").read_text(encoding="utf-8")
