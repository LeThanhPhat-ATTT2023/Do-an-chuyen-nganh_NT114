import json
from pathlib import Path

import numpy as np
import pytest
import torch

from graphslm_ids.offline.federated.config import (
    DEFAULT_FL_CONFIG,
    apply_overrides,
    get_dotted,
    load_fl_config,
    parse_set_args,
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
