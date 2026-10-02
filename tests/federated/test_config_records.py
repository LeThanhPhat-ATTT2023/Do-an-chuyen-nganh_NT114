import copy
import datetime
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from graphslm_ids.offline.federated.config import (
    DEFAULT_FL_CONFIG,
    apply_overrides,
    get_dotted,
    load_fl_config,
    parse_set_args,
    set_dotted,
    unknown_keys,
    yaml_load,
)
from graphslm_ids.offline.federated.records import append_jsonl, to_jsonable, write_json


def test_load_default_and_yaml_merge(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("model:\n  hidden_dim: 32\nattack:\n  name: alie\n", encoding="utf-8")
    cfg = load_fl_config(path)
    assert cfg["model"]["hidden_dim"] == 32
    assert cfg["model"]["num_layers"] == DEFAULT_FL_CONFIG["model"]["num_layers"]
    assert cfg["attack"]["name"] == "alie"
    assert cfg["defense"]["aggregator"]["name"] == "fedavg"


def test_dotted_overrides_and_set_args():
    overrides = parse_set_args(
        ["attack.malicious_ratio=0.3", "defense.aggregator={name: median}", "new.key=none"]
    )
    cfg = apply_overrides(load_fl_config(None), overrides)
    assert cfg["attack"]["malicious_ratio"] == 0.3
    assert cfg["defense"]["aggregator"] == {"name": "median"}
    assert cfg["new"]["key"] == "none"
    assert get_dotted(cfg, "attack.malicious_ratio") == 0.3
    assert get_dotted(cfg, "missing.x", 7) == 7


def test_parse_set_args_requires_equals():
    with pytest.raises(ValueError):
        parse_set_args(["attack.name"])


def test_defaults_are_not_mutated():
    cfg = load_fl_config(None)
    cfg["model"]["hidden_dim"] = 1
    assert DEFAULT_FL_CONFIG["model"]["hidden_dim"] != 1


def test_load_rejects_non_mapping(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_fl_config(path)


def test_to_jsonable_handles_numpy_torch_nan_and_paths(tmp_path):
    payload = {
        1: np.int64(3),
        "f": np.float32(0.5),
        "nan": float("nan"),
        "t": torch.tensor([1.0, 2.0]),
        "p": Path("a/b"),
        "b": np.bool_(True),
        "nested": [(np.arange(2),)],
    }
    assert to_jsonable(payload) == {
        "1": 3,
        "f": 0.5,
        "nan": None,
        "t": [1.0, 2.0],
        "p": str(Path("a/b")),
        "b": True,
        "nested": [[[0, 1]]],
    }
    write_json(tmp_path / "x.json", payload)
    assert json.loads((tmp_path / "x.json").read_text(encoding="utf-8"))["1"] == 3
    append_jsonl(tmp_path / "r.jsonl", {"a": 1})
    append_jsonl(tmp_path / "r.jsonl", {"a": 2})
    lines = (tmp_path / "r.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["a"] for line in lines] == [1, 2]


# --- YAML parsing: scientific floats and date strings -------------------------------------


def test_set_args_parse_scientific_floats_and_keep_dates_as_strings():
    parsed = parse_set_args(
        ["a=1e-3", "b=5E-5", "c=-2.5e+3", "d=.5", "e=2026-10-02", "f=3", "g=none"]
    )
    assert parsed["a"] == 0.001 and isinstance(parsed["a"], float)
    assert parsed["b"] == 5e-5 and isinstance(parsed["b"], float)
    assert parsed["c"] == -2500.0 and isinstance(parsed["c"], float)
    assert parsed["d"] == 0.5 and isinstance(parsed["d"], float)
    assert parsed["e"] == "2026-10-02"
    assert parsed["f"] == 3 and isinstance(parsed["f"], int)
    assert parsed["g"] == "none"


def test_yaml_file_reads_scientific_float_and_date_string(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("client: {lr: 1e-3}\nexperiment:\n  name: 2026-10-02\n", encoding="utf-8")
    cfg = load_fl_config(path)
    assert cfg["client"]["lr"] == 0.001 and isinstance(cfg["client"]["lr"], float)
    assert cfg["experiment"]["name"] == "2026-10-02"
    assert yaml_load("lr: 5E-5") == {"lr": 5e-5}


def test_fl_yaml_loader_does_not_mutate_global_safe_loader():
    yaml_load("x: 1e-3")  # make sure the FL loader has been exercised first
    assert yaml.safe_load("2026-10-02") == datetime.date(2026, 10, 2)
    assert yaml.safe_load("1e-3") == "1e-3"


# --- unknown-key detection ----------------------------------------------------------------


def test_unknown_keys_flags_typos_but_not_open_subtrees_or_non_dict_defaults():
    assert unknown_keys(load_fl_config(None)) == []
    overrides = parse_set_args(
        [
            "attack.params.z=1",
            "sampler.fanouts.anything=3",
            "experiment.labels.partition=x",
            "defense.aggregator.params.f=2",
            "client.adamw_betas=[0.8, 0.9]",
            "data.clean_eval_labels=labels.npy",
        ]
    )
    cfg = apply_overrides(load_fl_config(None), overrides)
    assert unknown_keys(cfg) == []
    cfg = apply_overrides(cfg, parse_set_args(["attack.malicous_ratio=0.1"]))
    assert unknown_keys(cfg) == ["attack.malicous_ratio"]


def test_unknown_keys_reports_nested_and_top_level_paths_sorted():
    cfg = load_fl_config(None)
    cfg["client"]["gcl"]["wieght"] = 1
    cfg["bogus"] = {"a": 1}
    cfg["attack"]["params"]["fine"] = {"deep": 1}
    assert unknown_keys(cfg) == ["bogus", "client.gcl.wieght"]


# --- set_dotted / apply_overrides hardening ------------------------------------------------


@pytest.mark.parametrize("key", ["", "a..b", ".a", "a."])
def test_set_dotted_rejects_empty_key_or_segment(key):
    with pytest.raises(ValueError):
        set_dotted({}, key, 1)


def test_parse_set_args_rejects_empty_key():
    with pytest.raises(ValueError):
        parse_set_args(["=5"])
    with pytest.raises(ValueError):
        apply_overrides(load_fl_config(None), {"": 5})


def test_set_dotted_rejects_non_mapping_intermediate():
    cfg = load_fl_config(None)
    with pytest.raises(ValueError):
        set_dotted(cfg, "client.adamw_betas.0", 0.5)
    assert cfg["client"]["adamw_betas"] == [0.9, 0.95]
    with pytest.raises(ValueError):
        apply_overrides(cfg, {"model.hidden_dim.x": 1})


def test_set_dotted_replaces_none_intermediate_with_dict():
    cfg = load_fl_config(None)
    set_dotted(cfg, "data.clean_eval_labels.path", "x.npy")
    assert cfg["data"]["clean_eval_labels"] == {"path": "x.npy"}


def test_apply_overrides_does_not_mutate_input_or_alias_values():
    base = load_fl_config(None)
    snapshot = copy.deepcopy(base)
    value = {"name": "median"}
    out = apply_overrides(base, {"attack.name": "alie", "defense.aggregator": value})
    assert base == snapshot
    assert out["attack"]["name"] == "alie"
    value["name"] = "mutated"
    assert out["defense"]["aggregator"] == {"name": "median"}


# --- deep_update edge cases -----------------------------------------------------------------


def test_empty_yaml_section_keeps_default_dict(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("attack:\n", encoding="utf-8")
    cfg = load_fl_config(path)
    assert cfg["attack"] == DEFAULT_FL_CONFIG["attack"]
    assert cfg["attack"] is not DEFAULT_FL_CONFIG["attack"]


def test_null_override_of_non_dict_default_still_sets_none(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("data:\n  graph_npz: null\n  clean_eval_labels: null\n", encoding="utf-8")
    cfg = load_fl_config(path)
    assert cfg["data"]["graph_npz"] is None
    assert cfg["data"]["clean_eval_labels"] is None
    assert cfg["data"]["splits_json"] == DEFAULT_FL_CONFIG["data"]["splits_json"]


def test_three_level_yaml_merge_keeps_sibling_defaults(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("client: {gcl: {enabled: true}}\n", encoding="utf-8")
    cfg = load_fl_config(path)
    assert cfg["client"]["gcl"]["enabled"] is True
    assert cfg["client"]["gcl"]["weight"] == 0.3
    assert cfg["client"]["lr"] == DEFAULT_FL_CONFIG["client"]["lr"]


# --- write_json ------------------------------------------------------------------------------


def test_write_json_creates_parents_and_leaves_no_tmp_file(tmp_path):
    target = tmp_path / "not" / "yet" / "x.json"
    write_json(target, {"k": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"k": 1}
    assert target.read_text(encoding="utf-8").endswith("\n")
    assert [p.name for p in target.parent.iterdir()] == ["x.json"]


def test_write_json_failure_keeps_previous_file_and_leaves_no_tmp_file(tmp_path):
    target = tmp_path / "x.json"
    write_json(target, {"k": 1})
    with pytest.raises(TypeError):
        write_json(target, {"bad": object()})
    assert json.loads(target.read_text(encoding="utf-8")) == {"k": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]
