# FL-HGT Poisoning Testbed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a federated-learning simulation of the GraphSLM HGT flow classifier with client-local graphs, five model-poisoning attacks, seven robust aggregators, three delta-only screeners, baselines and a resumable experiment-matrix runner.

**Architecture:** New package `src/graphslm_ids/offline/federated/`. Delta-space modules (`linalg`, `attacks`, `aggregators`, `screening`, `server`) work on `(N, D)` float32 tensors only and never import HGT/graph code. Graph-side modules (`partition`, `client_graph`, `params`, `local_trainer`, `simulation`) reuse the existing trainer building blocks (`to_torch_batch`, `make_neighbor_loader`, loss helpers, `EACSController`, `HeteroNeighborSampler`) without modifying `train_hgt_flow_classifier.py`.

**Tech Stack:** Python 3.13, PyTorch 2.11 (CPU locally, CUDA on the L40S), NumPy, SciPy (`norm.ppf`), scikit-learn (`MeanShift`), PyYAML, psutil, pytest.

**Spec:** `docs/superpowers/specs/2026-10-02-fl-hgt-poisoning-testbed-design.md`

## Conventions (read once)

- Work from the worktree root `D:\Tai_lieu_nam_ba\Hoc_ky_2\Do_an_chuyen_nganh\Do-an-chuyen-nganh_NT114\.claude\worktrees\fl-poisoning-testbed`.
- Interpreter: `D:\v\nt114\Scripts\python.exe` (written `PY` below). Tests: `PY -m pytest tests/federated/<file> -q`. `tests/conftest.py` already puts this worktree's `src/` first on `sys.path`.
- The venv has an editable install pointing at the MAIN checkout's `src/`. Anything run outside pytest from the worktree needs `PYTHONPATH=src` (scripts under `scripts/fl/` insert it themselves).
- `docs/superpowers/` is gitignored but tracked: add files there with `git add -f`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Never touch `src/graphslm_ids/offline/training/train_hgt_flow_classifier.py`.

## File map

| File | Responsibility |
|---|---|
| `src/graphslm_ids/offline/federated/__init__.py` | package marker |
| `.../federated/config.py` | `DEFAULT_FL_CONFIG`, `load_fl_config`, dotted overrides, `--set` parsing |
| `.../federated/records.py` | `to_jsonable`, `write_json`, `append_jsonl` |
| `.../federated/linalg.py` | pairwise distances/cosines, chunked coordinate median / trimmed mean, robust z, SVD |
| `.../federated/partition.py` | `ClientPartition`, IID / Dirichlet split, stratified subsample |
| `.../federated/client_graph.py` | `packet_owner`, `build_client_artifact`, `recompute_host_features` |
| `.../federated/params.py` | `ParamLayout` flatten/unflatten, HGT parameter groups, active groups |
| `.../federated/attacks.py` | `AttackContext`, sign-flip, ALIE, Min-Max, Min-Sum, ScaleSign, `build_attack` |
| `.../federated/aggregators.py` | `clip_deltas`, FedAvg, Median, TrimmedMean, MultiKrum, SignGuard, MSGuard, DnC, relation-aware wrapper |
| `.../federated/screening.py` | NoScreen, TemporalAudit, MultiSignal, BrayCurtis, `build_screener` |
| `.../federated/server.py` | `ClientMessage`, `FLServer` (integrity → clip → screen → aggregate) |
| `.../federated/metrics.py` | confusion-matrix eval loop, IDS metrics, detection counts, convergence |
| `.../federated/local_trainer.py` | `ClientState`, `LocalTrainer`, local class weights, round LR |
| `.../federated/simulation.py` | `load_context`, `run_experiment` (fl / centralized / local_only) |
| `.../federated/cli.py` | `python -m graphslm_ids.offline.federated.cli` |
| `.../federated/matrix.py` | `expand_grid`, `run_grid`, `summarize`, CSV/markdown writers |
| `scripts/fl/run_matrix.py`, `scripts/fl/summarize_matrix.py` | thin CLIs over `matrix.py` |
| `configs/fl/*.yaml` | screen / full tiers, matrix grids, baselines grid |
| `tests/federated/*` | fixture + unit/smoke tests |

---

### Task 1: Package skeleton, config and records

**Files:**
- Create: `src/graphslm_ids/offline/federated/__init__.py`
- Create: `src/graphslm_ids/offline/federated/config.py`
- Create: `src/graphslm_ids/offline/federated/records.py`
- Create: `tests/federated/__init__.py`
- Test: `tests/federated/test_config_records.py`

- [ ] **Step 1: Write the failing tests**

`tests/federated/__init__.py` is an empty file. `tests/federated/test_config_records.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PY -m pytest tests/federated/test_config_records.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'graphslm_ids.offline.federated'`

- [ ] **Step 3: Write the implementation**

`src/graphslm_ids/offline/federated/__init__.py`:

```python
"""Federated-learning poisoning testbed for the HGT flow classifier.

Design: docs/superpowers/specs/2026-10-02-fl-hgt-poisoning-testbed-design.md
"""
```

`src/graphslm_ids/offline/federated/config.py`:

```python
"""FL experiment configuration: defaults, YAML loading and dotted overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_FAMILIES: dict[str, list[str]] = {
    "web": ["CommandInjection", "XSS", "SqlInjection", "Uploading_Attack", "BrowserHijacking"],
    "ddos": [
        "DDoS-ACK_Fragmentation",
        "DDoS-ICMP_Flood",
        "DDoS-ICMP_Fragmentation",
        "DDoS-PSHACK_Flood",
        "DDoS-RSTFINFlood",
    ],
    "recon": [
        "Recon-HostDiscovery",
        "Recon-OSScan",
        "Recon-PingSweep",
        "Recon-PortScan",
        "VulnerabilityScan",
    ],
    "mirai": ["Mirai-udpplain"],
    "malware": ["Backdoor_Malware"],
}

DEFAULT_FL_CONFIG: dict[str, Any] = {
    "experiment": {"name": "fl_run", "labels": {}},
    "data": {
        "artifact_version": "v3",
        "graph_npz": "outputs/v3_ob/graph.npz",
        "graph_meta_json": "outputs/v3_ob/graph.meta.json",
        "splits_json": "outputs/v3_ob/splits.json",
        "split_protocol": "random",
        "packet_dtype": "preserve",
        "add_reverse_edges": True,
        "use_semantic_edge_weights": True,
        "standardize_flow_features": True,
        "clean_eval_labels": None,
    },
    "model": {
        "hidden_dim": 64,
        "num_layers": 2,
        "num_heads": 4,
        "dropout": 0.2,
        "ffn_multiplier": 2,
    },
    "sampler": {
        "hops": 2,
        "fanouts": {},
        "reverse_fanouts": {},
        "always_include_all_techniques": True,
        "always_include_all_tactics": True,
    },
    "partition": {
        "scheme": "dirichlet",
        "num_clients": 10,
        "alpha": 0.5,
        "min_flows_per_client": 50,
        "max_flows_per_client": None,
        "seed": 42,
    },
    "client": {
        "local_epochs": 1,
        "max_local_steps": None,
        "batch_seed_flows": 256,
        "lr": 1.0e-3,
        "lr_schedule": "cosine_rounds",
        "lr_min": 1.0e-5,
        "weight_decay": 5.0e-5,
        "adamw_eps": 1.0e-6,
        "adamw_betas": [0.9, 0.95],
        "grad_clip_norm": 1.0,
        "optimizer_state": "reset",
        "loss_type": "focal",
        "focal_gamma": 1.5,
        "label_smoothing": 0.05,
        "class_weight": "balanced",
        "class_weight_method": "cb",
        "cb_beta": 0.985,
        "class_weight_cap": 4.0,
        "stratified_batch_sampling": True,
        "drop_edge_prob": 0.1,
        "num_workers": 0,
        "pin_memory": False,
        "gcl": {
            "enabled": False,
            "weight": 0.3,
            "temperature": 0.1,
            "n_negatives": 16,
            "class_technique_map_csv": "data/mitre/class_technique_map.csv",
        },
        "eacs": {
            "enabled": False,
            "anchor_mask_npy": None,
            "warmup_rounds": 5,
            "ema_decay": 0.9,
            "lambda_disambig": 0.7,
            "suspect_classes": ["CommandInjection", "XSS", "SqlInjection", "Uploading_Attack"],
            "mitre_dir": "data/mitre",
        },
    },
    "federated": {
        "mode": "fl",
        "rounds": 30,
        "participation": 1.0,
        "server_lr": 1.0,
        "seed": 42,
        "cache_clients": True,
    },
    "attack": {
        "name": "none",
        "malicious_ratio": 0.0,
        "knowledge": "omniscient",
        "start_round": 1,
        "params": {},
    },
    "defense": {
        "clip": {"mode": "none", "factor": 1.0, "value": None},
        "screen": {"name": "none", "params": {}},
        "aggregator": {"name": "fedavg", "params": {}},
        "assumed_malicious_ratio": 0.2,
        "relation_aware": False,
        "weighting": "samples",
    },
    "eval": {
        "every": 1,
        "max_val_flows": None,
        "monitor": "val_macro_f1",
        "batch_seed_flows": 512,
        "families": copy.deepcopy(DEFAULT_FAMILIES),
    },
    "output": {"dir": "outputs/fl/fl_run", "save_checkpoints": True},
    "device": "auto",
}


def deep_update(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_fl_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return copy.deepcopy(DEFAULT_FL_CONFIG)
    with Path(path).open("r", encoding="utf-8") as fh:
        user = yaml.safe_load(fh) or {}
    if not isinstance(user, dict):
        raise ValueError(f"FL config {path} must be a YAML mapping")
    return deep_update(DEFAULT_FL_CONFIG, user)


def set_dotted(config: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    node = config
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def get_dotted(config: dict[str, Any], key: str, default: Any = None) -> Any:
    node: Any = config
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def parse_set_args(items: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        out[key.strip()] = yaml.safe_load(raw)
    return out


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    out = copy.deepcopy(config)
    for key, value in (overrides or {}).items():
        set_dotted(out, key, copy.deepcopy(value))
    return out
```

`src/graphslm_ids/offline/federated/records.py`:

```python
"""JSON-safe conversion and small file writers for FL run records."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(value) for value in obj]
    if isinstance(obj, torch.Tensor):
        return to_jsonable(obj.detach().cpu().tolist())
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def append_jsonl(path: str | Path, record: Any) -> None:
    with Path(path).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(to_jsonable(record)) + "\n")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PY -m pytest tests/federated/test_config_records.py -q`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/graphslm_ids/offline/federated/__init__.py src/graphslm_ids/offline/federated/config.py src/graphslm_ids/offline/federated/records.py tests/federated/__init__.py tests/federated/test_config_records.py
git commit -m "feat(fl): add FL config defaults, dotted overrides and JSON records"
```

---

### Task 2: Dense delta helpers (`linalg.py`)

**Files:**
- Create: `src/graphslm_ids/offline/federated/linalg.py`
- Test: `tests/federated/test_linalg.py`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest
import torch

from graphslm_ids.offline.federated.linalg import (
    coordinate_median,
    coordinate_trimmed_mean,
    cosine_to,
    mean_excluding_self,
    median_excluding_self,
    pairwise_cosine,
    pairwise_sq_dists,
    robust_z,
    top_right_singular_vector,
)


def test_pairwise_sq_dists_matches_cdist():
    torch.manual_seed(0)
    x = torch.randn(5, 7)
    ref = torch.cdist(x.double(), x.double()).pow(2)
    assert torch.allclose(pairwise_sq_dists(x), ref, atol=1e-4)


def test_pairwise_cosine_and_cosine_to():
    x = torch.tensor([[1.0, 0.0], [0.0, 2.0], [-3.0, 0.0]])
    c = pairwise_cosine(x)
    assert c[0, 1].item() == pytest.approx(0.0, abs=1e-9)
    assert c[0, 2].item() == pytest.approx(-1.0)
    assert torch.allclose(c.diagonal(), torch.ones(3, dtype=torch.float64))
    got = cosine_to(x, torch.tensor([1.0, 0.0]))
    assert torch.allclose(got, torch.tensor([1.0, 0.0, -1.0], dtype=torch.float64))


def test_coordinate_median_and_trimmed_mean_are_chunk_invariant():
    torch.manual_seed(1)
    x = torch.randn(6, 11)
    assert torch.allclose(coordinate_median(x, chunk=4), x.sort(0).values[2:4].mean(0))
    x5 = torch.randn(5, 11)
    assert torch.allclose(coordinate_median(x5, chunk=3), x5.median(0).values)
    assert torch.allclose(coordinate_trimmed_mean(x, 1, chunk=4), x.sort(0).values[1:5].mean(0))
    with pytest.raises(ValueError):
        coordinate_trimmed_mean(x, 3)


def test_robust_z_and_offdiag_helpers():
    z = robust_z([1.0, 2.0, 3.0, 100.0])
    assert z[3] > 10 and abs(z[1]) < 1
    m = np.array([[1.0, 0.2, 0.4], [0.2, 1.0, 0.6], [0.4, 0.6, 1.0]])
    assert median_excluding_self(m).tolist() == pytest.approx([0.3, 0.4, 0.5])
    assert mean_excluding_self(m).tolist() == pytest.approx([0.3, 0.4, 0.5])
    assert median_excluding_self(np.ones((1, 1))).tolist() == [1.0]


def test_top_right_singular_vector_finds_dominant_axis():
    x = torch.zeros(4, 3)
    x[:, 1] = torch.tensor([3.0, -3.0, 2.0, -2.0])
    v = top_right_singular_vector(x)
    assert abs(float(v[1])) == pytest.approx(1.0, abs=1e-5)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PY -m pytest tests/federated/test_linalg.py -q`
Expected: FAIL — `ModuleNotFoundError: ... federated.linalg`

- [ ] **Step 3: Write the implementation**

```python
"""Dense helpers on (N, D) client-delta matrices.

N (clients per round) is small; D (HGT parameters, ~6.5M for the full model) is
large, so coordinate-wise reductions run over column chunks to bound memory.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch

CHUNK = 1 << 20


def pairwise_sq_dists(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    out = torch.zeros((n, n), dtype=torch.float64, device=x.device)
    for i in range(n):
        out[i] = (x - x[i]).pow(2).sum(dim=1).to(torch.float64)
    return out


def pairwise_cosine(x: torch.Tensor) -> torch.Tensor:
    gram = (x @ x.T).to(torch.float64)
    norms = gram.diagonal().clamp_min(1e-24).sqrt()
    return gram / (norms[:, None] * norms[None, :])


def cosine_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    ref = ref.to(x.device, x.dtype)
    dots = (x @ ref).to(torch.float64)
    denom = x.norm(dim=1).to(torch.float64) * float(ref.norm())
    return dots / denom.clamp_min(1e-24)


def _sorted_chunks(x: torch.Tensor, chunk: int) -> Iterator[tuple[int, torch.Tensor]]:
    for start in range(0, x.shape[1], chunk):
        yield start, x[:, start:start + chunk].sort(dim=0).values


def coordinate_median(x: torch.Tensor, chunk: int = CHUNK) -> torch.Tensor:
    n = x.shape[0]
    out = torch.empty(x.shape[1], dtype=x.dtype, device=x.device)
    for start, s in _sorted_chunks(x, chunk):
        end = start + s.shape[1]
        out[start:end] = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    return out


def coordinate_trimmed_mean(x: torch.Tensor, k: int, chunk: int = CHUNK) -> torch.Tensor:
    n = x.shape[0]
    if k < 0 or 2 * k >= n:
        raise ValueError(f"cannot trim {k} per side from {n} rows")
    out = torch.empty(x.shape[1], dtype=x.dtype, device=x.device)
    for start, s in _sorted_chunks(x, chunk):
        out[start:start + s.shape[1]] = s[k:n - k].mean(dim=0)
    return out


def top_right_singular_vector(x: torch.Tensor) -> torch.Tensor:
    _u, _s, vh = torch.linalg.svd(x.to(torch.float32), full_matrices=False)
    return vh[0]


def robust_center_scale(
    values: np.ndarray | list[float], axis: int = 0, floor: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(values, dtype=np.float64)
    center = np.median(v, axis=axis)
    scale = 1.4826 * np.median(np.abs(v - center), axis=axis)
    return center, np.maximum(scale, floor)


def robust_z(values: np.ndarray | list[float], floor: float = 1e-6) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    center, scale = robust_center_scale(v, axis=0, floor=floor)
    return (v - center) / scale


def _offdiag_rows(mat: np.ndarray) -> list[np.ndarray]:
    m = np.asarray(mat, dtype=np.float64)
    mask = ~np.eye(m.shape[0], dtype=bool)
    return [m[i][mask[i]] for i in range(m.shape[0])]


def median_excluding_self(mat: np.ndarray) -> np.ndarray:
    return np.array([float(np.median(r)) if r.size else 1.0 for r in _offdiag_rows(mat)])


def mean_excluding_self(mat: np.ndarray) -> np.ndarray:
    return np.array([float(r.mean()) if r.size else 1.0 for r in _offdiag_rows(mat)])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PY -m pytest tests/federated/test_linalg.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/graphslm_ids/offline/federated/linalg.py tests/federated/test_linalg.py
git commit -m "feat(fl): add chunked dense helpers for client-delta matrices"
```

---

### Task 3: Client partitioning (`partition.py`)

**Files:**
- Create: `src/graphslm_ids/offline/federated/partition.py`
- Test: `tests/federated/test_partition.py`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from graphslm_ids.offline.federated.partition import (
    ClientPartition,
    make_partition,
    partition_dirichlet,
    partition_iid,
    subsample_partition,
)


def _labels(n=600, c=4, seed=0):
    return np.random.default_rng(seed).integers(0, c, n)


def test_iid_is_deterministic_disjoint_and_complete():
    train = np.arange(100, 400)
    a = partition_iid(train, 5, seed=1)
    b = partition_iid(train, 5, seed=1)
    assert all(np.array_equal(x, y) for x, y in zip(a.client_flow_ids, b.client_flow_ids))
    assert np.array_equal(a.all_flow_ids(), train)
    assert sum(a.sizes()) == train.size
    assert max(a.sizes()) - min(a.sizes()) <= 1


def test_dirichlet_small_alpha_is_more_skewed_than_large_alpha():
    labels = _labels()
    train = np.arange(labels.size)

    def skew(alpha):
        part = partition_dirichlet(train, labels, 5, alpha, seed=3, min_flows_per_client=1)
        hist = part.class_histogram(labels, 4).astype(float)
        frac = hist / hist.sum(axis=1, keepdims=True)
        return float(frac.max(axis=1).mean())

    assert skew(0.1) > skew(100.0)


def test_dirichlet_keeps_a_disjoint_cover_of_train():
    labels = _labels()
    train = np.arange(0, labels.size, 2)
    part = partition_dirichlet(train, labels, 4, 0.5, seed=0, min_flows_per_client=5)
    assert np.array_equal(part.all_flow_ids(), np.sort(train))
    assert min(part.sizes()) >= 5


def test_dirichlet_raises_when_minimum_is_unreachable():
    labels = _labels(40)
    with pytest.raises(ValueError, match="alpha"):
        partition_dirichlet(np.arange(40), labels, 4, 0.5, seed=0, min_flows_per_client=30)


def test_subsample_keeps_every_class_and_respects_cap():
    labels = _labels()
    part = partition_iid(np.arange(labels.size), 2, seed=0)
    sub = subsample_partition(part, labels, 40, seed=0)
    assert max(sub.sizes()) <= 40
    for orig, small in zip(part.client_flow_ids, sub.client_flow_ids):
        assert set(small.tolist()) <= set(orig.tolist())
        assert set(labels[small].tolist()) == set(labels[orig].tolist())


def test_make_partition_dispatch_and_errors():
    labels = _labels()
    part = make_partition(np.arange(labels.size), labels, {"scheme": "iid", "num_clients": 3, "seed": 0, "max_flows_per_client": 50})
    assert part.num_clients == 3 and max(part.sizes()) <= 50
    with pytest.raises(ValueError):
        make_partition(np.arange(10), np.zeros(10, dtype=int), {"scheme": "bogus"})


def test_json_round_trip():
    part = partition_iid(np.arange(20), 3, seed=0)
    back = ClientPartition.from_json(part.to_json())
    assert back.scheme == "iid" and back.seed == 0
    assert all(np.array_equal(x, y) for x, y in zip(part.client_flow_ids, back.client_flow_ids))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PY -m pytest tests/federated/test_partition.py -q`
Expected: FAIL — `ModuleNotFoundError: ... federated.partition`

- [ ] **Step 3: Write the implementation**

```python
"""Split TRAIN flows into simulated FL clients (IID or Dirichlet label skew)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ClientPartition:
    client_flow_ids: list[np.ndarray]
    scheme: str
    alpha: float | None
    seed: int

    @property
    def num_clients(self) -> int:
        return len(self.client_flow_ids)

    def sizes(self) -> list[int]:
        return [int(ids.shape[0]) for ids in self.client_flow_ids]

    def all_flow_ids(self) -> np.ndarray:
        if not self.client_flow_ids:
            return np.empty((0,), dtype=np.int64)
        return np.sort(np.concatenate(self.client_flow_ids))

    def class_histogram(self, labels: np.ndarray, num_classes: int) -> np.ndarray:
        labels = np.asarray(labels, dtype=np.int64)
        return np.stack(
            [np.bincount(labels[ids], minlength=num_classes) for ids in self.client_flow_ids]
        ).astype(np.int64)

    def to_json(self) -> dict[str, Any]:
        return {
            "scheme": self.scheme,
            "alpha": self.alpha,
            "seed": self.seed,
            "client_flow_ids": [ids.astype(np.int64).tolist() for ids in self.client_flow_ids],
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "ClientPartition":
        return cls(
            client_flow_ids=[np.asarray(ids, dtype=np.int64) for ids in payload["client_flow_ids"]],
            scheme=str(payload["scheme"]),
            alpha=None if payload.get("alpha") is None else float(payload["alpha"]),
            seed=int(payload["seed"]),
        )


def partition_iid(train_ids: np.ndarray, num_clients: int, seed: int) -> ClientPartition:
    if num_clients < 1:
        raise ValueError("num_clients must be >= 1")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(np.asarray(train_ids, dtype=np.int64))
    return ClientPartition([np.sort(c) for c in np.array_split(perm, num_clients)], "iid", None, int(seed))


def partition_dirichlet(
    train_ids: np.ndarray,
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    min_flows_per_client: int = 1,
    max_retries: int = 100,
) -> ClientPartition:
    if num_clients < 1:
        raise ValueError("num_clients must be >= 1")
    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be > 0")
    train_ids = np.asarray(train_ids, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int64)[train_ids]
    classes = np.unique(y)
    for attempt in range(max_retries):
        rng = np.random.default_rng(seed + attempt)
        buckets: list[list[np.ndarray]] = [[] for _ in range(num_clients)]
        for c in classes:
            ids_c = train_ids[y == c].copy()
            rng.shuffle(ids_c)
            props = rng.dirichlet(np.full(num_clients, float(alpha)))
            cuts = (np.cumsum(props)[:-1] * ids_c.shape[0]).astype(np.int64)
            for k, part in enumerate(np.split(ids_c, cuts)):
                if part.size:
                    buckets[k].append(part)
        clients = [
            np.sort(np.concatenate(b)) if b else np.empty((0,), dtype=np.int64) for b in buckets
        ]
        if min(c.shape[0] for c in clients) >= min_flows_per_client:
            return ClientPartition(clients, "dirichlet", float(alpha), int(seed))
    raise ValueError(
        f"Dirichlet partition (alpha={alpha}) could not give every client >= "
        f"{min_flows_per_client} flows after {max_retries} attempts; increase alpha "
        "or lower min_flows_per_client"
    )


def _stratified_take(ids: np.ndarray, labels: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    y = labels[ids]
    classes, counts = np.unique(y, return_counts=True)
    quota = np.maximum(1, np.floor(counts * (k / ids.shape[0])).astype(np.int64))
    quota = np.minimum(quota, counts)
    while quota.sum() > k:
        quota[int(np.argmax(quota))] -= 1
    picked = [
        rng.choice(ids[y == c], size=int(q), replace=False)
        for c, q in zip(classes, quota)
        if q > 0
    ]
    return np.sort(np.concatenate(picked)).astype(np.int64)


def subsample_partition(
    partition: ClientPartition, labels: np.ndarray, max_flows_per_client: int, seed: int
) -> ClientPartition:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=np.int64)
    cap = int(max_flows_per_client)
    out = [
        ids if ids.shape[0] <= cap else _stratified_take(ids, labels, cap, rng)
        for ids in partition.client_flow_ids
    ]
    return ClientPartition(out, partition.scheme, partition.alpha, partition.seed)


def make_partition(train_ids: np.ndarray, labels: np.ndarray, cfg: dict[str, Any]) -> ClientPartition:
    scheme = str(cfg.get("scheme", "iid")).lower()
    num_clients = int(cfg.get("num_clients", 10))
    seed = int(cfg.get("seed", 42))
    if scheme == "iid":
        part = partition_iid(train_ids, num_clients, seed)
    elif scheme == "dirichlet":
        part = partition_dirichlet(
            train_ids,
            labels,
            num_clients,
            float(cfg.get("alpha", 0.5)),
            seed,
            int(cfg.get("min_flows_per_client", 1)),
        )
    else:
        raise ValueError(f"unknown partition scheme {scheme!r}; expected 'iid' or 'dirichlet'")
    cap = cfg.get("max_flows_per_client")
    if cap:
        part = subsample_partition(part, labels, int(cap), seed)
    return part
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PY -m pytest tests/federated/test_partition.py -q`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add src/graphslm_ids/offline/federated/partition.py tests/federated/test_partition.py
git commit -m "feat(fl): add IID/Dirichlet client partitioning with stratified subsample"
```

---

### Task 4: Tiny v3 fixture and client graphs (`client_graph.py`)

**Files:**
- Create: `tests/federated/conftest.py`
- Create: `src/graphslm_ids/offline/federated/client_graph.py`
- Test: `tests/federated/test_client_graph.py`

- [ ] **Step 1: Write the fixture and the failing tests**

`tests/federated/conftest.py`:

```python
"""Tiny synthetic v3 artifact shared by the federated tests (CPU, well under 1 s)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

LABELS = {"Benign": 0, "XSS": 1, "DDoS-ICMP_Flood": 2}
FLOW_FEATURES = ["proto", "src_port", "dst_port", "sig_a", "sig_b", "sig_c"]
TECHNIQUES = {"T1059": 0, "T1190": 1, "T1498": 2, "T1595": 3}


def write_tiny_v3(root: Path, num_flows: int = 48, seed: int = 0) -> dict[str, Path]:
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    y = np.arange(num_flows, dtype=np.int64) % 3
    flows = np.arange(num_flows, dtype=np.int64)
    flow_x = rng.normal(size=(num_flows, len(FLOW_FEATURES))).astype(np.float32)
    flow_x[:, 0] = 6.0
    flow_x[:, 1] = rng.integers(1024, 65535, size=num_flows)
    flow_x[:, 2] = np.where(y == 1, 80, np.where(y == 2, 0, 443))
    flow_x[:, 3] += 3.0 * y
    num_packets = 2 * num_flows
    packet_x = rng.normal(size=(num_packets, 8)).astype(np.float16)
    packet_x[:, 0] += np.repeat(y, 2).astype(np.float16)
    burst = np.concatenate(
        [np.stack([flows[:-1], flows[1:]]), np.stack([flows[:-2], flows[2:]])], axis=1
    )
    xss_pk = 2 * flows[y == 1]
    ddos_pk = 2 * flows[y == 2]
    xss_flows = flows[y == 1]
    arrays = {
        "flow_x": flow_x,
        "flow_y": y,
        "packet_x": packet_x,
        "host_x": np.zeros((6, 4), dtype=np.float32),
        "technique_x": rng.normal(size=(len(TECHNIQUES), 5)).astype(np.float32),
        "contain_edge_index": np.stack([np.repeat(flows, 2), np.arange(num_packets)]),
        "link_edge_index": np.stack([2 * flows, 2 * flows + 1]),
        "link_edge_attr": np.full((num_flows, 1), 0.01, dtype=np.float32),
        "from_host_edge_index": np.stack([flows, flows % 3]),
        "from_host_edge_attr": np.ones((num_flows, 1), dtype=np.float32),
        "to_host_edge_index": np.stack([flows, 3 + (flows % 3)]),
        "to_host_edge_attr": np.ones((num_flows, 1), dtype=np.float32),
        "burst_neighbor_edge_index": burst,
        "burst_neighbor_edge_attr": np.ones((burst.shape[1], 1), dtype=np.float32),
        "evidence_injection_edge_index": np.stack([xss_pk, np.zeros_like(xss_pk)]),
        "evidence_injection_edge_attr": np.full((xss_pk.size, 1), 0.9, dtype=np.float32),
        "evidence_recon_edge_index": np.stack([ddos_pk, np.full_like(ddos_pk, 3)]),
        "evidence_recon_edge_attr": np.full((ddos_pk.size, 1), 0.7, dtype=np.float32),
        "flow_technique_edge_index": np.stack([xss_flows, np.ones_like(xss_flows)]),
        "flow_technique_edge_attr": np.full((xss_flows.size, 1), 0.8, dtype=np.float32),
        "has_subtechnique_edge_index": np.array([[0], [1]], dtype=np.int64),
        "technique_tactic_edge_index": np.array([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=np.int64),
        "technique_tactic_edge_attr": np.ones((4, 1), dtype=np.float32),
    }
    npz = root / "graph.npz"
    np.savez(npz, **arrays)
    meta = root / "graph.meta.json"
    meta.write_text(
        json.dumps(
            {
                "artifact_version": "v3",
                "num_tactics": 2,
                "num_techniques": len(TECHNIQUES),
                "label_mapping": LABELS,
                "flow_feature_names": FLOW_FEATURES,
                "technique_id_to_idx": TECHNIQUES,
            }
        ),
        encoding="utf-8",
    )
    perm = rng.permutation(num_flows)
    n_hold = num_flows // 6
    split = {
        "train": sorted(perm[2 * n_hold:].tolist()),
        "val": sorted(perm[:n_hold].tolist()),
        "test": sorted(perm[n_hold:2 * n_hold].tolist()),
    }
    splits = root / "splits.json"
    splits.write_text(json.dumps({"random": split, "temporal": split}), encoding="utf-8")
    xss = np.flatnonzero(y == 1)
    clean = y.copy()
    clean[xss[:2]] = 0
    np.save(root / "clean_eval_labels.npy", clean)
    anchor = np.zeros(num_flows, dtype=bool)
    anchor[xss[2:]] = True
    np.save(root / "eacs_anchor_mask.npy", anchor)
    ctm = root / "class_technique_map.csv"
    ctm.write_text(
        "class,technique,weight,note\nXSS,T1059,1.0,\nDDoS-ICMP_Flood,T1498,1.0,\nBenign,,,\n",
        encoding="utf-8",
    )
    return {
        "graph_npz": npz,
        "graph_meta_json": meta,
        "splits_json": splits,
        "clean_eval_labels": root / "clean_eval_labels.npy",
        "anchor_mask_npy": root / "eacs_anchor_mask.npy",
        "class_technique_map_csv": ctm,
    }


@pytest.fixture
def tiny_paths(tmp_path):
    return write_tiny_v3(tmp_path / "tiny")


@pytest.fixture
def tiny_artifact(tiny_paths):
    from graphslm_ids.offline.training.hetero_graph_artifact import load_v3_artifact

    return load_v3_artifact(
        tiny_paths["graph_npz"], tiny_paths["graph_meta_json"], packet_dtype="float32"
    )
```

`tests/federated/test_client_graph.py`:

```python
import numpy as np

from graphslm_ids.offline.federated.client_graph import (
    FROM_HOST,
    TO_HOST,
    build_client_artifact,
    packet_owner,
    recompute_host_features,
)
from graphslm_ids.offline.training.hetero_graph_artifact import HeteroGraphArtifact


def test_client_graph_keeps_only_owned_flow_and_packet_edges(tiny_artifact):
    art = tiny_artifact
    client = np.arange(0, art.flow_y.shape[0], 2)
    sub = build_client_artifact(art, client)
    owner = packet_owner(art)
    flows = set(client.tolist())
    assert sub.edge_index.keys() == art.edge_index.keys()
    for key, edges in sub.edge_index.items():
        for pos, ntype in ((0, key[0]), (1, key[2])):
            ids = edges[pos]
            if ntype == "flow":
                assert set(ids.tolist()) <= flows, key
            if ntype == "packet":
                assert set(owner[ids].tolist()) <= flows, key
        assert sub.edge_attr[key].shape[0] == edges.shape[1]
    burst = sub.edge_index[("flow", "burst_neighbor", "flow")]
    assert burst.shape[1] > 0 and np.all(burst % 2 == 0)


def test_client_graph_keeps_mitre_knowledge_edges(tiny_artifact):
    sub = build_client_artifact(tiny_artifact, np.array([0]))
    for key in (
        ("technique", "has_subtechnique", "technique"),
        ("technique", "belongs_to_tactic", "tactic"),
        ("tactic", "rev_belongs_to_tactic", "technique"),
    ):
        assert sub.edge_index[key].shape[1] == tiny_artifact.edge_index[key].shape[1]


def test_client_graph_shares_large_arrays(tiny_artifact):
    sub = build_client_artifact(tiny_artifact, np.arange(5))
    assert sub.node_features["packet"] is tiny_artifact.node_features["packet"]
    assert sub.node_features["flow"] is tiny_artifact.node_features["flow"]
    assert sub.flow_y is tiny_artifact.flow_y


def test_recompute_host_features_matches_hand_count():
    flow_x = np.array([[6, 1000, 80], [6, 1001, 80], [6, 1002, 22]], dtype=np.float32)
    art = HeteroGraphArtifact(
        node_features={
            "flow": flow_x,
            "packet": np.zeros((0, 2), dtype=np.float32),
            "host": np.full((3, 4), 99.0, dtype=np.float32),
            "technique": np.zeros((1, 2), dtype=np.float32),
            "tactic": np.arange(1)[:, None],
        },
        edge_index={
            FROM_HOST: np.array([[0, 1, 2], [0, 0, 1]]),
            TO_HOST: np.array([[0, 1, 2], [1, 2, 2]]),
        },
        edge_attr={},
        flow_y=np.zeros(3, dtype=np.int64),
        metadata={"flow_feature_names": ["proto", "src_port", "dst_port"]},
    )
    hx = recompute_host_features(art, np.array([0, 1]))
    assert hx.tolist() == [[2, 0, 1, 2], [0, 1, 0, 0], [0, 1, 0, 0]]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PY -m pytest tests/federated/test_client_graph.py -q`
Expected: FAIL — `ModuleNotFoundError: ... federated.client_graph`

- [ ] **Step 3: Write the implementation**

```python
"""Per-client graph view: keep only edges whose flow/packet endpoints the client owns.

Global node ids are preserved, so EACS masks, clean labels and splits stay valid.
Technique/tactic nodes are public MITRE knowledge and are replicated as-is.
"""

from __future__ import annotations

import logging

import numpy as np

from graphslm_ids.offline.training.hetero_graph_artifact import HeteroGraphArtifact

_LOG = logging.getLogger(__name__)

CONTAINS = ("flow", "contains", "packet")
FROM_HOST = ("flow", "from_host", "host")
TO_HOST = ("flow", "to_host", "host")


def packet_owner(artifact: HeteroGraphArtifact) -> np.ndarray:
    num_packets = int(artifact.node_features["packet"].shape[0])
    owner = np.full((num_packets,), -1, dtype=np.int64)
    edges = np.asarray(artifact.edge_index.get(CONTAINS, np.zeros((2, 0))), dtype=np.int64)
    if edges.shape[1]:
        owner[edges[1]] = edges[0]
    return owner


def _flow_to_host(artifact: HeteroGraphArtifact, key: tuple[str, str, str], num_flows: int) -> np.ndarray:
    out = np.full((num_flows,), -1, dtype=np.int64)
    edges = np.asarray(artifact.edge_index.get(key, np.zeros((2, 0))), dtype=np.int64)
    if edges.shape[1]:
        out[edges[0]] = edges[1]
    return out


def recompute_host_features(artifact: HeteroGraphArtifact, client_flow_ids: np.ndarray) -> np.ndarray:
    """out-degree, in-degree, #distinct dst ports, #distinct dst hosts over the
    client's flows only (graph_builder._build_host_tier aggregates over ALL flows)."""
    host_x = np.asarray(artifact.node_features["host"])
    if host_x.ndim != 2 or host_x.shape[1] != 4:
        _LOG.warning("host_x has shape %s (expected (H, 4)); keeping global host features", host_x.shape)
        return np.array(host_x, dtype=np.float32, copy=True)
    num_hosts = int(host_x.shape[0])
    num_flows = int(artifact.node_features["flow"].shape[0])
    ids = np.asarray(client_flow_ids, dtype=np.int64)
    src = _flow_to_host(artifact, FROM_HOST, num_flows)[ids]
    dst = _flow_to_host(artifact, TO_HOST, num_flows)[ids]
    out = np.zeros((num_hosts, 4), dtype=np.float32)
    has_src = src >= 0
    has_dst = dst >= 0
    out[:, 0] = np.bincount(src[has_src], minlength=num_hosts)
    out[:, 1] = np.bincount(dst[has_dst], minlength=num_hosts)
    names = list((artifact.metadata or {}).get("flow_feature_names") or [])
    if "dst_port" in names:
        col = names.index("dst_port")
        ports = np.asarray(artifact.node_features["flow"][ids, col]).astype(np.int64)
        pairs = np.unique(np.stack([src[has_src], ports[has_src]], axis=1), axis=0)
        out[:, 2] = np.bincount(pairs[:, 0], minlength=num_hosts)
    else:
        _LOG.warning("flow_feature_names has no 'dst_port'; host feature 2 set to 0")
    both = has_src & has_dst
    pairs = np.unique(np.stack([src[both], dst[both]], axis=1), axis=0)
    out[:, 3] = np.bincount(pairs[:, 0], minlength=num_hosts)
    return out


def build_client_artifact(
    artifact: HeteroGraphArtifact,
    client_flow_ids: np.ndarray,
    *,
    owner: np.ndarray | None = None,
    recompute_host: bool = True,
) -> HeteroGraphArtifact:
    ids = np.unique(np.asarray(client_flow_ids, dtype=np.int64))
    owner = packet_owner(artifact) if owner is None else np.asarray(owner, dtype=np.int64)
    num_flows = int(artifact.node_features["flow"].shape[0])
    flow_mask = np.zeros((num_flows,), dtype=bool)
    flow_mask[ids] = True
    packet_mask = np.zeros(owner.shape, dtype=bool)
    has_owner = owner >= 0
    packet_mask[has_owner] = flow_mask[owner[has_owner]]
    owned = {"flow": flow_mask, "packet": packet_mask}

    edge_index: dict[tuple[str, str, str], np.ndarray] = {}
    edge_attr: dict[tuple[str, str, str], np.ndarray] = {}
    for key, edges in artifact.edge_index.items():
        edges = np.asarray(edges, dtype=np.int64)
        src_type, _relation, dst_type = key
        keep = np.ones((edges.shape[1],), dtype=bool)
        if src_type in owned:
            keep &= owned[src_type][edges[0]]
        if dst_type in owned:
            keep &= owned[dst_type][edges[1]]
        edge_index[key] = edges[:, keep]
        attr = artifact.edge_attr.get(key)
        if attr is not None:
            edge_attr[key] = np.asarray(attr)[keep]

    node_features = dict(artifact.node_features)
    if recompute_host and "host" in node_features:
        node_features["host"] = recompute_host_features(artifact, ids)
    return HeteroGraphArtifact(
        node_features=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        flow_y=artifact.flow_y,
        metadata=artifact.metadata,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PY -m pytest tests/federated/test_client_graph.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/federated/conftest.py src/graphslm_ids/offline/federated/client_graph.py tests/federated/test_client_graph.py
git commit -m "feat(fl): build client-local graph views and local host features"
```

---

### Task 5: Parameter layout and HGT groups (`params.py`)

**Files:**
- Create: `src/graphslm_ids/offline/federated/params.py`
- Test: `tests/federated/test_params.py`

- [ ] **Step 1: Write the failing tests**

```python
import pytest
import torch

from graphslm_ids.models.hgt import HeteroGraphTransformer
from graphslm_ids.offline.federated.params import ParamLayout, group_of


def _tiny_model():
    edge_types = [
        ("flow", "contains", "packet"),
        ("packet", "rev_contains", "flow"),
        ("flow", "burst_neighbor", "flow"),
    ]
    return HeteroGraphTransformer(
        node_input_dims={"flow": 4, "packet": 3, "technique": 2},
        edge_types=edge_types,
        num_classes=3,
        num_tactics=2,
        hidden_dim=8,
        num_layers=2,
        num_heads=2,
        node_types=["flow", "packet", "technique", "tactic"],
    )


def test_flatten_unflatten_round_trip():
    model = _tiny_model()
    layout = ParamLayout.from_model(model)
    vec = layout.flatten(model.state_dict())
    assert vec.shape == (layout.numel,) and vec.dtype == torch.float32
    rebuilt = layout.unflatten(vec)
    for key, value in model.state_dict().items():
        assert torch.equal(rebuilt[key], value)
    _tiny_model().load_state_dict(rebuilt)
    with pytest.raises(ValueError):
        layout.unflatten(torch.zeros(layout.numel + 1))


def test_groups_cover_relations_nodes_and_heads():
    layout = ParamLayout.from_model(_tiny_model())
    assert {
        "rel:flow__contains__packet",
        "rel:packet__rev_contains__flow",
        "rel:flow__burst_neighbor__flow",
        "node:flow",
        "node:packet",
        "classifier",
        "tactic_embedding",
    } <= set(layout.group_names)
    covered = sum(b - a for g in layout.group_names for a, b in layout.groups[g])
    assert covered == layout.numel


def test_group_of_examples():
    assert group_of("layers.3.relation_prior.flow__contains__packet") == "rel:flow__contains__packet"
    assert group_of("layers.0.relation_key.a__b__c.weight") == "rel:a__b__c"
    assert group_of("layers.0.ffn.host.0.weight") == "node:host"
    assert group_of("input_projection.packet.0.bias") == "node:packet"
    assert group_of("classifier.2.weight") == "classifier"


def test_active_groups_detect_untouched_relation():
    layout = ParamLayout.from_model(_tiny_model())
    deltas = torch.randn(2, layout.numel)
    cols = layout.group_coords(["rel:flow__burst_neighbor__flow"])
    deltas[1, cols] = 0.0
    active = layout.active_groups(deltas)
    g = layout.group_names.index("rel:flow__burst_neighbor__flow")
    assert bool(active[0, g]) and not bool(active[1, g])
    total = layout.group_norms(deltas)[0].double().pow(2).sum()
    assert total.item() == pytest.approx(deltas[0].double().pow(2).sum().item(), rel=1e-5)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PY -m pytest tests/federated/test_params.py -q`
Expected: FAIL — `ModuleNotFoundError: ... federated.params`

- [ ] **Step 3: Write the implementation**

```python
"""Flatten/unflatten HGT weights and map coordinates to HGT parameter groups.

Groups: ``rel:<src__rel__dst>`` (relation_key/value/prior across layers),
``node:<type>`` (input projection + per-type q/k/v/out/norm/ffn), ``tactic_embedding``,
``classifier``. A relation absent from every batch of a client round gets grad None,
AdamW skips it, so its delta group is exactly zero ("inactive").
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import torch

_REL_RE = re.compile(r"^layers\.\d+\.relation_(?:key|value|prior)\.([^.]+)")
_NODE_RE = re.compile(
    r"^(?:input_projection|layers\.\d+\.(?:query|key|value|out|norm_attn|norm_ffn|ffn))\.([^.]+)"
)


def group_of(name: str) -> str:
    match = _REL_RE.match(name)
    if match:
        return f"rel:{match.group(1)}"
    match = _NODE_RE.match(name)
    if match:
        return f"node:{match.group(1)}"
    if name.startswith("tactic_embedding"):
        return "tactic_embedding"
    if name.startswith("classifier"):
        return "classifier"
    return "other"


@dataclass
class ParamLayout:
    names: list[str]
    shapes: list[torch.Size]
    dtypes: list[torch.dtype]
    offsets: list[int]
    numel: int
    groups: dict[str, list[tuple[int, int]]] = field(default_factory=dict)

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, torch.Tensor]) -> "ParamLayout":
        names: list[str] = []
        shapes: list[torch.Size] = []
        dtypes: list[torch.dtype] = []
        offsets: list[int] = []
        groups: dict[str, list[tuple[int, int]]] = {}
        offset = 0
        for name, tensor in state_dict.items():
            if not torch.is_floating_point(tensor):
                raise ValueError(f"non-float state entry {name!r} is not supported by ParamLayout")
            n = int(tensor.numel())
            names.append(name)
            shapes.append(tensor.shape)
            dtypes.append(tensor.dtype)
            offsets.append(offset)
            groups.setdefault(group_of(name), []).append((offset, offset + n))
            offset += n
        return cls(names, shapes, dtypes, offsets, offset, groups)

    @classmethod
    def from_model(cls, model: torch.nn.Module) -> "ParamLayout":
        return cls.from_state_dict(model.state_dict())

    @property
    def group_names(self) -> list[str]:
        return list(self.groups.keys())

    def flatten(self, state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [state_dict[name].detach().reshape(-1).to("cpu", torch.float32) for name in self.names]
        )

    def unflatten(self, vec: torch.Tensor) -> dict[str, torch.Tensor]:
        if vec.numel() != self.numel:
            raise ValueError(f"vector has {vec.numel()} entries, layout expects {self.numel}")
        vec = vec.detach().to("cpu", torch.float32)
        out: dict[str, torch.Tensor] = {}
        for name, shape, dtype, start in zip(self.names, self.shapes, self.dtypes, self.offsets):
            n = math.prod(int(d) for d in shape)
            out[name] = vec[start:start + n].view(shape).to(dtype).clone()
        return out

    def group_coords(self, group_names: list[str]) -> torch.Tensor:
        parts = [torch.arange(a, b) for g in group_names for a, b in self.groups[g]]
        return torch.cat(parts) if parts else torch.empty(0, dtype=torch.long)

    def group_norms(self, deltas: torch.Tensor) -> torch.Tensor:
        cols = []
        for g in self.group_names:
            sq = torch.zeros(deltas.shape[0], dtype=torch.float64, device=deltas.device)
            for a, b in self.groups[g]:
                sq += deltas[:, a:b].to(torch.float64).pow(2).sum(dim=1)
            cols.append(sq.sqrt())
        return torch.stack(cols, dim=1).to(torch.float32)

    def active_groups(self, deltas: torch.Tensor) -> torch.Tensor:
        return self.group_norms(deltas) > 0
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PY -m pytest tests/federated/test_params.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/graphslm_ids/offline/federated/params.py tests/federated/test_params.py
git commit -m "feat(fl): add ParamLayout with HGT relation/node parameter groups"
```

---

### Task 6: Model-poisoning attacks (`attacks.py`)

**Files:**
- Create: `src/graphslm_ids/offline/federated/attacks.py`
- Test: `tests/federated/test_attacks.py`

- [ ] **Step 1: Write the failing tests**

```python
import pytest
import torch

from graphslm_ids.offline.federated.attacks import (
    ALIE,
    AttackContext,
    MinMax,
    MinSum,
    ScaleSign,
    SignFlip,
    alie_z,
    build_attack,
    krum_sign_counts,
    match_sign_counts,
    search_gamma,
)
from graphslm_ids.offline.federated.linalg import pairwise_sq_dists


def _ctx(benign, honest, knowledge="omniscient"):
    return AttackContext(
        honest=honest,
        benign=benign,
        num_participants=benign.shape[0] + honest.shape[0],
        knowledge=knowledge,
    )


def test_alie_z_matches_paper_example():
    assert alie_z(50, 24) == pytest.approx(1.75, abs=0.01)
    assert alie_z(10, 2) == pytest.approx(0.2533, abs=1e-3)


def test_sign_flip_negates_honest_deltas():
    honest = torch.randn(2, 5)
    out = SignFlip(scale=2.0).craft(_ctx(torch.randn(3, 5), honest), 2)
    assert torch.allclose(out, -2.0 * honest)


def test_alie_uses_benign_mean_and_std():
    benign = torch.randn(8, 6)
    out = ALIE(z=1.5).craft(_ctx(benign, torch.zeros(2, 6)), 2)
    expected = benign.mean(0) + 1.5 * benign.std(0, unbiased=False)
    assert out.shape == (2, 6)
    assert torch.allclose(out[0], expected) and torch.allclose(out[1], expected)


def test_partial_knowledge_uses_attacker_rows():
    benign = torch.randn(8, 6)
    honest = torch.randn(3, 6) + 5.0
    out = ALIE(z=0.0).craft(_ctx(benign, honest, "partial"), 3)
    assert torch.allclose(out[0], honest.mean(0))


def test_search_gamma_finds_threshold():
    assert search_gamma(lambda g: g <= 3.7, 10.0, 1e-6) == pytest.approx(3.7, abs=1e-4)


@pytest.mark.parametrize("cls", [MinMax, MinSum])
def test_gamma_attacks_sit_on_their_constraint_boundary(cls):
    torch.manual_seed(0)
    benign = torch.randn(10, 40)
    attack = cls(perturbation="std", gamma_init=10.0, tol=1e-6)
    v = attack.craft(_ctx(benign, torch.zeros(0, 40)), 3)[0]
    d = pairwise_sq_dists(benign)

    def measure(vec):
        dist = (benign - vec).pow(2).sum(1).double()
        return float(dist.max()) if cls is MinMax else float(dist.sum())

    bound = float(d.max()) if cls is MinMax else float(d.sum(1).max())
    assert attack.last_gamma > 0
    assert measure(v) <= bound * (1 + 1e-4)
    mu, p = benign.mean(0), -benign.std(0, unbiased=False)
    assert measure(mu + 1.01 * attack.last_gamma * p) > bound


def test_match_sign_counts_handles_both_directions():
    out = match_sign_counts(torch.tensor([3.0, 1.0, -2.0, -0.5, 0.0]), 1, 3)
    assert int((out > 0).sum()) == 1 and int((out < 0).sum()) == 3
    out2 = match_sign_counts(torch.tensor([-1.0, -2.0, 0.0, 0.0]), 2, 1)
    assert int((out2 > 0).sum()) == 2 and int((out2 < 0).sum()) == 1


def test_scale_sign_matches_target_sign_counts():
    torch.manual_seed(1)
    benign = torch.randn(8, 200)
    honest = torch.randn(3, 200)
    attack = ScaleSign(gamma_init=10.0, tol=1e-4)
    out = attack.craft(_ctx(benign, honest), 3)
    pos, _zero, neg = krum_sign_counts(honest)
    assert int((out[0] > 0).sum()) == pos and int((out[0] < 0).sum()) == neg
    assert attack.last_gamma >= 1.0
    assert torch.equal(out[0], out[2])


def test_build_attack_validates_names_and_params():
    assert build_attack({"name": "alie", "params": {"z": 1.0}}).z == 1.0
    assert build_attack({"name": "none"}).name == "none"
    with pytest.raises(ValueError):
        build_attack({"name": "bogus"})
    with pytest.raises(ValueError):
        build_attack({"name": "min_max", "params": {"bogus": 1}})
    with pytest.raises(ValueError):
        build_attack({"name": "min_max", "params": {"perturbation": "bogus"}})


def test_reference_needs_rows_and_valid_knowledge():
    with pytest.raises(ValueError):
        AttackContext(torch.zeros(0, 3), torch.zeros(0, 3), 0).reference()
    with pytest.raises(ValueError):
        AttackContext(torch.zeros(1, 3), torch.zeros(1, 3), 2, knowledge="bogus").reference()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PY -m pytest tests/federated/test_attacks.py -q`
Expected: FAIL — `ModuleNotFoundError: ... federated.attacks`

- [ ] **Step 3: Write the implementation**

```python
"""Model-poisoning attacks on flattened deltas (experiment-only Byzantine hook).

``knowledge="omniscient"``: the reference set is the round's benign deltas (worst
case). ``"partial"``: only the attackers' own honest deltas. ALIE / Min-Max / Min-Sum /
ScaleSign make every attacker send one shared vector (collusion by construction).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from scipy.stats import norm as _normal

from graphslm_ids.offline.federated.linalg import pairwise_sq_dists

KNOWLEDGE = ("omniscient", "partial")


@dataclass
class AttackContext:
    honest: torch.Tensor
    benign: torch.Tensor
    num_participants: int
    knowledge: str = "omniscient"

    def reference(self) -> torch.Tensor:
        if self.knowledge not in KNOWLEDGE:
            raise ValueError(f"unknown attack knowledge {self.knowledge!r}; expected one of {KNOWLEDGE}")
        if self.knowledge == "omniscient" and self.benign.shape[0] > 0:
            return self.benign
        if self.honest.shape[0] > 0:
            return self.honest
        if self.benign.shape[0] > 0:
            return self.benign
        raise ValueError("attack has neither benign nor honest reference deltas")


def search_gamma(
    oracle: Callable[[float], bool], gamma_init: float = 10.0, tol: float = 1e-5, max_iter: int = 200
) -> float:
    """Largest gamma with oracle(gamma) True — NDSS'21 Alg. 1 with the official
    code's step schedule (first move is +-gamma_init/2)."""
    gamma = float(gamma_init)
    step = float(gamma_init)
    succ = 0.0
    for _ in range(max_iter):
        if abs(succ - gamma) <= tol:
            break
        if oracle(gamma):
            succ = gamma
            gamma += step / 2.0
        else:
            gamma -= step / 2.0
        step /= 2.0
    return succ


def alie_z(num_participants: int, num_malicious: int) -> float:
    """Baruch et al. 2019, Alg. 3: s = floor(n/2 + 1) - m; z = Phi^-1((n - s) / n)."""
    n = int(num_participants)
    s = math.floor(n / 2 + 1) - int(num_malicious)
    q = min(max((n - s) / n, 1e-6), 1.0 - 1e-6)
    return float(_normal.ppf(q))


def perturbation_direction(ref: torch.Tensor, kind: str) -> torch.Tensor:
    mu = ref.mean(dim=0)
    if kind == "std":
        return -ref.std(dim=0, unbiased=False)
    if kind == "unit":
        return -mu / mu.norm().clamp_min(1e-12)
    if kind == "sign":
        return -torch.sign(mu)
    raise ValueError(f"unknown perturbation {kind!r}; expected std, unit or sign")


def sign_counts(x: torch.Tensor) -> tuple[int, int, int]:
    return int((x > 0).sum()), int((x == 0).sum()), int((x < 0).sum())


def krum_sign_counts(honest: torch.Tensor) -> tuple[int, int, int]:
    """ScaleSign Eq. 10: the attacker sign-count triple closest (Krum-style) to the rest."""
    counts = torch.tensor([sign_counts(row) for row in honest], dtype=torch.float64)
    if counts.shape[0] == 1:
        return tuple(int(v) for v in counts[0].tolist())
    d = torch.cdist(counts, counts).pow(2)
    j = int(d.sum(dim=1).argmin())
    return tuple(int(v) for v in counts[j].tolist())


def match_sign_counts(g: torch.Tensor, target_pos: int, target_neg: int) -> torch.Tensor:
    """ScaleSign Alg. 4 sign repair: zero the smallest surplus entries, or promote the
    entries nearest zero to the boundary value of the missing sign."""
    g = g.clone()
    pos_idx = torch.nonzero(g > 0).flatten()
    p_now = int(pos_idx.numel())
    if target_pos < p_now:
        sel = torch.topk(g[pos_idx], p_now - target_pos, largest=False).indices
        g[pos_idx[sel]] = 0.0
    elif target_pos > p_now:
        tau = float(g[pos_idx].min()) if p_now else 1e-12
        nonpos = torch.nonzero(g <= 0).flatten()
        k = min(target_pos - p_now, int(nonpos.numel()))
        sel = torch.topk(g[nonpos], k, largest=True).indices
        g[nonpos[sel]] = tau
    neg_idx = torch.nonzero(g < 0).flatten()
    n_now = int(neg_idx.numel())
    if target_neg < n_now:
        sel = torch.topk(g[neg_idx], n_now - target_neg, largest=True).indices
        g[neg_idx[sel]] = 0.0
    elif target_neg > n_now:
        tau = float(g[neg_idx].max()) if n_now else -1e-12
        zero_idx = torch.nonzero(g == 0).flatten()
        k = min(target_neg - n_now, int(zero_idx.numel()))
        g[zero_idx[:k]] = tau
    return g


class Attack:
    name = "base"
    needs_honest = False

    def craft(self, ctx: AttackContext, num_malicious: int) -> torch.Tensor:
        raise NotImplementedError


class NoAttack(Attack):
    name = "none"
    needs_honest = True

    def craft(self, ctx: AttackContext, num_malicious: int) -> torch.Tensor:
        return ctx.honest.clone()


class SignFlip(Attack):
    name = "sign_flip"
    needs_honest = True

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = float(scale)

    def craft(self, ctx: AttackContext, num_malicious: int) -> torch.Tensor:
        if ctx.honest.shape[0] != num_malicious:
            raise ValueError("sign_flip needs one honest delta per attacker")
        return -self.scale * ctx.honest


class ALIE(Attack):
    name = "alie"

    def __init__(self, z: float | None = None) -> None:
        self.z = None if z is None else float(z)

    def craft(self, ctx: AttackContext, num_malicious: int) -> torch.Tensor:
        ref = ctx.reference()
        z = self.z if self.z is not None else alie_z(ctx.num_participants, num_malicious)
        vec = ref.mean(dim=0) + z * ref.std(dim=0, unbiased=False)
        return vec.unsqueeze(0).repeat(num_malicious, 1)


class _GammaAttack(Attack):
    def __init__(self, perturbation: str = "std", gamma_init: float = 10.0, tol: float = 1e-5) -> None:
        if perturbation not in {"std", "unit", "sign"}:
            raise ValueError(f"unknown perturbation {perturbation!r}; expected std, unit or sign")
        self.perturbation = perturbation
        self.gamma_init = float(gamma_init)
        self.tol = float(tol)
        self.last_gamma = 0.0

    def _oracle(
        self, a: torch.Tensor, b: torch.Tensor, cc: float, dists: torch.Tensor
    ) -> Callable[[float], bool]:
        raise NotImplementedError

    def craft(self, ctx: AttackContext, num_malicious: int) -> torch.Tensor:
        ref = ctx.reference()
        mu = ref.mean(dim=0)
        p = perturbation_direction(ref, self.perturbation)
        diff = mu.unsqueeze(0) - ref
        # ||mu + g p - r_i||^2 = a_i + 2 g b_i + g^2 cc  -> each oracle call is O(N).
        a = diff.pow(2).sum(dim=1).to(torch.float64)
        b = (diff @ p).to(torch.float64)
        cc = float(p.pow(2).sum())
        oracle = self._oracle(a, b, cc, pairwise_sq_dists(ref))
        self.last_gamma = search_gamma(oracle, self.gamma_init, self.tol)
        return (mu + self.last_gamma * p).unsqueeze(0).repeat(num_malicious, 1)


class MinMax(_GammaAttack):
    """Shejwalkar & Houmansadr NDSS'21: max_i ||m - g_i|| <= max_ij ||g_i - g_j||."""

    name = "min_max"

    def _oracle(self, a, b, cc, dists):
        bound = float(dists.max())
        return lambda g: float((a + 2.0 * g * b + g * g * cc).max()) <= bound


class MinSum(_GammaAttack):
    """NDSS'21 Eq. 7 (max over i on the right-hand side, as in the paper)."""

    name = "min_sum"

    def _oracle(self, a, b, cc, dists):
        bound = float(dists.sum(dim=1).max())
        return lambda g: float((a + 2.0 * g * b + g * g * cc).sum()) <= bound


class ScaleSign(Attack):
    """Yang et al., TIFS 2025, §V. Gamma search uses the NDSS oracle instead of the
    printed Alg. 2 (which re-checks a stale g* and never moves on failure)."""

    name = "scale_sign"
    needs_honest = True

    def __init__(self, gamma_init: float = 10.0, tol: float = 1e-3, constraint: str = "min_max") -> None:
        if constraint not in {"min_max", "min_sum"}:
            raise ValueError(f"unknown scale_sign constraint {constraint!r}")
        self.gamma_init = float(gamma_init)
        self.tol = float(tol)
        self.constraint = constraint
        self.last_gamma = 1.0

    def craft(self, ctx: AttackContext, num_malicious: int) -> torch.Tensor:
        honest = ctx.honest
        if honest.shape[0] == 0:
            raise ValueError("scale_sign needs the attackers' honest deltas")
        gbar = honest.mean(dim=0)
        zero = torch.zeros_like(gbar)
        pos = torch.where(gbar > 0, gbar, zero)
        neg = torch.where(gbar < 0, gbar, zero)
        ref = ctx.reference()
        # ||g pos + neg/g - r_i||^2 with disjoint supports of pos and neg.
        aa = float(pos.pow(2).sum())
        bb = float(neg.pow(2).sum())
        rr = ref.pow(2).sum(dim=1).to(torch.float64)
        ar = (ref @ pos).to(torch.float64)
        br = (ref @ neg).to(torch.float64)
        dists = pairwise_sq_dists(ref)

        def sq(g: float) -> torch.Tensor:
            return g * g * aa + bb / (g * g) + rr - 2.0 * g * ar - (2.0 / g) * br

        if self.constraint == "min_max":
            bound = float(dists.max())
            ok = lambda g: g >= 1.0 and float(sq(g).max()) <= bound  # noqa: E731
        else:
            bound = float(dists.sum(dim=1).max())
            ok = lambda g: g >= 1.0 and float(sq(g).sum()) <= bound  # noqa: E731
        self.last_gamma = max(search_gamma(ok, self.gamma_init, self.tol), 1.0)
        crafted = self.last_gamma * pos + neg / self.last_gamma
        target_pos, _target_zero, target_neg = krum_sign_counts(honest)
        crafted = match_sign_counts(crafted, target_pos, target_neg)
        return crafted.unsqueeze(0).repeat(num_malicious, 1)


ATTACKS: dict[str, type[Attack]] = {
    "none": NoAttack,
    "sign_flip": SignFlip,
    "alie": ALIE,
    "min_max": MinMax,
    "min_sum": MinSum,
    "scale_sign": ScaleSign,
}


def build_attack(cfg: dict) -> Attack:
    name = str(cfg.get("name", "none")).lower()
    if name not in ATTACKS:
        raise ValueError(f"unknown attack {name!r}; choose from {sorted(ATTACKS)}")
    params = dict(cfg.get("params") or {})
    try:
        return ATTACKS[name](**params)
    except TypeError as exc:
        raise ValueError(f"bad params for attack {name!r}: {exc}") from exc
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PY -m pytest tests/federated/test_attacks.py -q`
Expected: `11 passed`

- [ ] **Step 5: Commit**

```bash
git add src/graphslm_ids/offline/federated/attacks.py tests/federated/test_attacks.py
git commit -m "feat(fl): add sign-flip, ALIE, Min-Max, Min-Sum and ScaleSign attacks"
```

---

### Task 7: Robust aggregators (`aggregators.py`)

**Files:**
- Create: `src/graphslm_ids/offline/federated/aggregators.py`
- Test: `tests/federated/test_aggregators.py`

- [ ] **Step 1: Write the failing tests**

```python
import pytest
import torch

from graphslm_ids.offline.federated.aggregators import (
    DnC,
    FedAvg,
    Median,
    MSGuard,
    MultiKrum,
    SignGuard,
    TrimmedMean,
    build_aggregator,
    clip_deltas,
    relation_aware_aggregate,
)
from graphslm_ids.offline.federated.params import ParamLayout


def _benign(n=10, d=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(d, generator=g)
    return base + 0.1 * torch.randn(n, d, generator=g)


def test_fedavg_matches_weighted_mean():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = FedAvg().aggregate(x, torch.tensor([1.0, 3.0])).delta
    assert torch.allclose(out, torch.tensor([2.5, 3.5]))


def test_median_odd_and_even():
    assert torch.allclose(Median().aggregate(torch.tensor([[1.0], [5.0], [3.0]]), torch.ones(3)).delta, torch.tensor([3.0]))
    assert torch.allclose(Median().aggregate(torch.tensor([[1.0], [2.0], [3.0], [10.0]]), torch.ones(4)).delta, torch.tensor([2.5]))


def test_trimmed_mean_drops_extremes_and_rejects_over_trim():
    x = torch.tensor([[0.0], [1.0], [2.0], [100.0]])
    assert torch.allclose(TrimmedMean(trim_k=1).aggregate(x, torch.ones(4)).delta, torch.tensor([1.5]))
    with pytest.raises(ValueError):
        TrimmedMean(trim_k=2).aggregate(x, torch.ones(4))


@pytest.mark.parametrize(
    "agg",
    [
        Median(),
        TrimmedMean(trim_ratio=0.2),
        MultiKrum(f_ratio=0.2),
        SignGuard(seed=0),
        MSGuard(seed=0, zeta=40),
        DnC(f_ratio=0.2, b=50, seed=0),
    ],
    ids=lambda a: a.name,
)
def test_robust_rules_resist_a_large_outlier(agg):
    x = _benign()
    x[0] = -50.0 * x[1]
    clean = x[1:].mean(0)
    fedavg = FedAvg().aggregate(x, torch.ones(10)).delta
    assert (fedavg - clean).norm() > clean.norm()
    out = agg.aggregate(x, torch.ones(10)).delta
    assert (out - clean).norm() < 0.5 * clean.norm()


@pytest.mark.parametrize("agg", [MultiKrum(f_ratio=0.2), SignGuard(seed=0), MSGuard(seed=0, zeta=40), DnC(f_ratio=0.2, b=50, seed=0)], ids=lambda a: a.name)
def test_filtering_rules_report_the_outlier_as_not_selected(agg):
    x = _benign()
    x[0] = -50.0 * x[1]
    result = agg.aggregate(x, torch.ones(10))
    assert result.selected is not None and not bool(result.selected[0])


def test_multi_krum_requires_enough_clients():
    with pytest.raises(ValueError):
        MultiKrum(f=2).aggregate(torch.randn(5, 4), torch.ones(5))


def test_clip_deltas_modes():
    x = torch.tensor([[3.0, 4.0], [0.0, 1.0], [0.0, 2.0]])
    clipped, norms = clip_deltas(x, "median", 1.0)
    assert torch.allclose(norms, torch.tensor([5.0, 1.0, 2.0]))
    assert float(clipped.norm(dim=1).max()) <= 2.0 + 1e-5
    same, _ = clip_deltas(x, "none")
    assert torch.equal(same, x)
    with pytest.raises(ValueError):
        clip_deltas(x, "fixed")
    with pytest.raises(ValueError):
        clip_deltas(x, "bogus")


def test_relation_aware_ignores_inactive_clients():
    layout = ParamLayout.from_state_dict(
        {"layers.0.relation_key.a__r__b.weight": torch.zeros(2), "classifier.0.weight": torch.zeros(2)}
    )
    deltas = torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 3.0, 3.0]])
    out = relation_aware_aggregate(FedAvg(), deltas, torch.ones(2), layout).delta
    assert torch.allclose(out, torch.tensor([1.0, 1.0, 2.0, 2.0]))
    with pytest.raises(ValueError):
        relation_aware_aggregate(MultiKrum(f=0), deltas, torch.ones(2), layout)


def test_build_aggregator_defaults_and_errors():
    agg = build_aggregator({"name": "trimmed_mean"}, assumed_malicious_ratio=0.3)
    assert isinstance(agg, TrimmedMean) and agg.trim_ratio == 0.3
    assert build_aggregator({"name": "multi_krum"}, assumed_malicious_ratio=0.1).f_ratio == 0.1
    with pytest.raises(ValueError):
        build_aggregator({"name": "nope"})
    with pytest.raises(ValueError):
        build_aggregator({"name": "fedavg", "params": {"bogus": 1}})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PY -m pytest tests/federated/test_aggregators.py -q`
Expected: FAIL — `ModuleNotFoundError: ... federated.aggregators`

- [ ] **Step 3: Write the implementation**

```python
"""Robust aggregation rules over (N, D) client deltas.

Weights carry ``num_samples * trust``. Filtering rules (Multi-Krum, SignGuard,
MSGuard, DnC) report which rows they kept in ``AggResult.selected``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from graphslm_ids.offline.federated.linalg import (
    coordinate_median,
    coordinate_trimmed_mean,
    cosine_to,
    median_excluding_self,
    pairwise_cosine,
    pairwise_sq_dists,
    top_right_singular_vector,
)
from graphslm_ids.offline.federated.params import ParamLayout


@dataclass
class AggResult:
    delta: torch.Tensor
    selected: torch.Tensor | None = None
    scores: dict[str, Any] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)


def weighted_mean(deltas: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    w = weights.to(deltas.device, deltas.dtype)
    total = float(w.sum())
    if total <= 0:
        raise ValueError("aggregation weights sum to zero")
    return (w / total) @ deltas


def clip_deltas(
    deltas: torch.Tensor, mode: str = "none", factor: float = 1.0, value: float | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    norms = deltas.norm(dim=1)
    if mode == "none":
        return deltas, norms
    if mode == "median":
        bound = float(factor) * float(norms.median())
    elif mode == "fixed":
        if value is None:
            raise ValueError("clip.mode=fixed requires clip.value")
        bound = float(value)
    else:
        raise ValueError(f"unknown clip mode {mode!r}; expected none, median or fixed")
    scale = (bound / norms.clamp_min(1e-12)).clamp(max=1.0)
    return deltas * scale.unsqueeze(1), norms


def _resolve_f(f: int | None, f_ratio: float, n: int) -> int:
    return int(f) if f is not None else int(math.floor(float(f_ratio) * n))


def largest_meanshift_cluster(feats: np.ndarray, quantile: float = 0.5) -> tuple[np.ndarray, bool]:
    """Boolean mask of the largest MeanShift cluster; (all True, True) on fallback."""
    from sklearn.cluster import MeanShift, estimate_bandwidth

    n = int(feats.shape[0])
    everyone = np.ones(n, dtype=bool)
    if n < 3:
        return everyone, True
    bandwidth = float(estimate_bandwidth(feats, quantile=quantile, n_samples=min(50, n)))
    if not np.isfinite(bandwidth) or bandwidth <= 1e-12:
        return everyone, True
    try:
        labels = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=False).fit(feats).labels_
    except ValueError:
        return everyone, True
    valid = labels >= 0
    if not valid.any():
        return everyone, True
    top = int(np.bincount(labels[valid]).argmax())
    return labels == top, False


def sign_features(deltas: torch.Tensor, ratio: float, rng: np.random.Generator) -> np.ndarray:
    """SignGuard features: fractions of positive / zero / negative coordinates on one
    contiguous random block (official code), each divided by its max over clients."""
    d = deltas.shape[1]
    block = max(1, int(round(ratio * d)))
    start = int(rng.integers(0, d - block + 1))
    sub = deltas[:, start:start + block]
    feats = torch.stack(
        [(sub > 0).double().mean(1), (sub == 0).double().mean(1), (sub < 0).double().mean(1)], dim=1
    ).cpu().numpy()
    peak = feats.max(axis=0)
    peak[peak <= 0] = 1.0
    return feats / peak


class Aggregator:
    name = "base"
    coordinatewise = False

    def aggregate(self, deltas: torch.Tensor, weights: torch.Tensor, ctx: dict | None = None) -> AggResult:
        raise NotImplementedError


class FedAvg(Aggregator):
    name = "fedavg"
    coordinatewise = True

    def aggregate(self, deltas, weights, ctx=None):
        return AggResult(weighted_mean(deltas, weights))


class Median(Aggregator):
    name = "median"
    coordinatewise = True

    def aggregate(self, deltas, weights, ctx=None):
        return AggResult(coordinate_median(deltas))


class TrimmedMean(Aggregator):
    name = "trimmed_mean"
    coordinatewise = True

    def __init__(self, trim_ratio: float = 0.2, trim_k: int | None = None) -> None:
        self.trim_ratio = float(trim_ratio)
        self.trim_k = None if trim_k is None else int(trim_k)

    def aggregate(self, deltas, weights, ctx=None):
        n = deltas.shape[0]
        k = self.trim_k if self.trim_k is not None else int(math.floor(self.trim_ratio * n))
        if 2 * k >= n:
            raise ValueError(f"trimmed_mean trims {k} per side but only {n} deltas were kept")
        return AggResult(coordinate_trimmed_mean(deltas, k), info={"trim_k": k})


class MultiKrum(Aggregator):
    name = "multi_krum"

    def __init__(self, f: int | None = None, f_ratio: float = 0.2, m: int | None = None) -> None:
        self.f = f
        self.f_ratio = float(f_ratio)
        self.m = m

    def aggregate(self, deltas, weights, ctx=None):
        weights = weights.to(deltas.device, torch.float32)
        n = deltas.shape[0]
        f = _resolve_f(self.f, self.f_ratio, n)
        if n <= 2 * f + 2:
            raise ValueError(f"multi_krum needs n > 2f + 2 (n={n}, f={f})")
        m = max(1, min(n - f if self.m is None else int(self.m), n))
        d2 = pairwise_sq_dists(deltas).cpu().numpy()
        remaining = list(range(n))
        selected: list[int] = []
        first_scores = np.full(n, np.nan)
        while len(selected) < m:
            r = len(remaining)
            k = max(1, r - f - 2)
            sub = d2[np.ix_(remaining, remaining)]
            scores = np.array([np.sort(np.delete(sub[i], i))[:k].sum() for i in range(r)])
            if not selected:
                first_scores[remaining] = scores
            selected.append(remaining.pop(int(np.argmin(scores))))
        mask = torch.zeros(n, dtype=torch.bool, device=deltas.device)
        mask[selected] = True
        return AggResult(
            weighted_mean(deltas[mask], weights[mask]),
            selected=mask,
            scores={"krum": first_scores},
            info={"f": f, "m": m},
        )


class SignGuard(Aggregator):
    """Xu et al. ICDCS'22 (+ official code): norm filter, sign-statistics MeanShift,
    median-norm clipped mean over the intersection. ``variant='sim'`` adds the cosine
    feature of SignGuard-Sim."""

    name = "signguard"

    def __init__(
        self,
        lower: float = 0.1,
        upper: float = 3.0,
        coord_ratio: float = 0.1,
        variant: str = "plain",
        quantile: float = 0.5,
        seed: int = 0,
    ) -> None:
        if variant not in {"plain", "sim"}:
            raise ValueError(f"unknown signguard variant {variant!r}")
        self.lower, self.upper = float(lower), float(upper)
        self.coord_ratio = float(coord_ratio)
        self.variant = variant
        self.quantile = float(quantile)
        self.rng = np.random.default_rng(seed)

    def _extra_features(self, deltas, ctx):
        if self.variant != "sim":
            return None
        prev = (ctx or {}).get("prev_aggregate")
        if prev is not None and float(prev.norm()) > 0:
            sim = cosine_to(deltas, prev).cpu().numpy()
        else:
            sim = median_excluding_self(pairwise_cosine(deltas).cpu().numpy())
        med = float(np.median(sim))
        return np.zeros_like(sim) if med <= 0 else np.clip(sim, 0.0, med) / (5.0 * med)

    def _features(self, deltas, ctx):
        feats = sign_features(deltas, self.coord_ratio, self.rng)
        extra = self._extra_features(deltas, ctx)
        return feats if extra is None else np.hstack([feats, extra[:, None]])

    def aggregate(self, deltas, weights, ctx=None):
        weights = weights.to(deltas.device, torch.float32)
        n = deltas.shape[0]
        norms = deltas.norm(dim=1)
        med = max(float(norms.median()), 1e-12)
        ratio = norms / med
        s1 = ((ratio >= self.lower) & (ratio <= self.upper)).cpu().numpy()
        s2, fallback = largest_meanshift_cluster(self._features(deltas, ctx), self.quantile)
        sel = s1 & s2
        if not sel.any():
            sel, fallback = s1, True
        if not sel.any():
            sel = np.ones(n, dtype=bool)
        mask = torch.as_tensor(sel, device=deltas.device)
        clip = torch.clamp(med / norms.clamp_min(1e-12), max=1.0)
        delta = weighted_mean(deltas[mask] * clip[mask].unsqueeze(1), weights[mask])
        return AggResult(
            delta,
            selected=mask,
            scores={"norm_ratio": ratio.cpu().numpy()},
            info={"fallback": bool(fallback), "median_norm": med},
        )


class MSGuard(SignGuard):
    """Yang et al. TIFS 2025 Alg. 8 (reconstructed: the paper gives no defaults, so
    SignGuard conventions are used). Features: sign stats + cosine score + spectral
    outlier score, each scaled to [0, 0.2] as in the paper."""

    name = "msguard"

    def __init__(
        self,
        lower: float = 0.1,
        upper: float = 3.0,
        coord_ratio: float = 0.1,
        zeta: int = 10000,
        quantile: float = 0.5,
        seed: int = 0,
    ) -> None:
        super().__init__(lower, upper, coord_ratio, "plain", quantile, seed)
        self.zeta = int(zeta)

    @staticmethod
    def cosine_score(deltas: torch.Tensor) -> np.ndarray:
        c = median_excluding_self(pairwise_cosine(deltas).cpu().numpy())
        theta, low = float(np.median(c)), float(c.min())
        out = np.full_like(c, 0.2)
        below = c < theta
        if theta > low:
            out[below] = (c[below] - low) / (theta - low) * 0.2
        return out

    def outlier_score(self, deltas: torch.Tensor) -> np.ndarray:
        mu = deltas.mean(dim=0)
        k = max(1, min(self.zeta // 2, deltas.shape[1] // 2))
        idx = torch.cat([torch.topk(mu, k).indices, torch.topk(-mu, k).indices]).unique()
        x = deltas[:, idx]
        x = x - x.mean(dim=0, keepdim=True)
        o = (x @ top_right_singular_vector(x)).pow(2).double().cpu().numpy()
        med, top = float(np.median(o)), float(o.max())
        out = np.zeros_like(o)
        above = o > med
        if top > med:
            out[above] = (o[above] - med) / (top - med) * 0.2
        return out

    def _features(self, deltas, ctx):
        return np.hstack(
            [
                sign_features(deltas, self.coord_ratio, self.rng),
                self.cosine_score(deltas)[:, None],
                self.outlier_score(deltas)[:, None],
            ]
        )


class DnC(Aggregator):
    """Divide-and-Conquer (NDSS'21 Alg. 2)."""

    name = "dnc"

    def __init__(
        self,
        niters: int = 1,
        c: float = 1.0,
        b: int = 10000,
        f: int | None = None,
        f_ratio: float = 0.2,
        seed: int = 0,
    ) -> None:
        self.niters, self.c, self.b = int(niters), float(c), int(b)
        self.f, self.f_ratio = f, float(f_ratio)
        self.rng = np.random.default_rng(seed)

    def aggregate(self, deltas, weights, ctx=None):
        weights = weights.to(deltas.device, torch.float32)
        n, d = deltas.shape
        f = _resolve_f(self.f, self.f_ratio, n)
        keep_n = max(1, n - int(math.floor(self.c * f)))
        good = np.ones(n, dtype=bool)
        for _ in range(self.niters):
            dims = np.sort(self.rng.choice(d, size=min(self.b, d), replace=False))
            x = deltas[:, torch.as_tensor(dims, device=deltas.device)]
            x = x - x.mean(dim=0, keepdim=True)
            s = (x @ top_right_singular_vector(x)).pow(2).cpu().numpy()
            keep = np.zeros(n, dtype=bool)
            keep[np.argsort(s, kind="stable")[:keep_n]] = True
            good &= keep
        if not good.any():
            good[:] = True
        mask = torch.as_tensor(good, device=deltas.device)
        return AggResult(weighted_mean(deltas[mask], weights[mask]), selected=mask, info={"f": f, "keep": keep_n})


_AGGREGATORS: dict[str, type[Aggregator]] = {
    "fedavg": FedAvg,
    "median": Median,
    "trimmed_mean": TrimmedMean,
    "multi_krum": MultiKrum,
    "signguard": SignGuard,
    "msguard": MSGuard,
    "dnc": DnC,
}


def build_aggregator(cfg: dict, *, assumed_malicious_ratio: float = 0.2, seed: int = 0) -> Aggregator:
    name = str(cfg.get("name", "fedavg")).lower()
    if name not in _AGGREGATORS:
        raise ValueError(f"unknown aggregator {name!r}; choose from {sorted(_AGGREGATORS)}")
    params = dict(cfg.get("params") or {})
    if name == "trimmed_mean":
        params.setdefault("trim_ratio", float(assumed_malicious_ratio))
    if name in {"multi_krum", "dnc"}:
        params.setdefault("f_ratio", float(assumed_malicious_ratio))
    if name in {"signguard", "msguard", "dnc"}:
        params.setdefault("seed", int(seed))
    try:
        return _AGGREGATORS[name](**params)
    except TypeError as exc:
        raise ValueError(f"bad params for aggregator {name!r}: {exc}") from exc


def relation_aware_aggregate(
    aggregator: Aggregator,
    deltas: torch.Tensor,
    weights: torch.Tensor,
    layout: ParamLayout,
    ctx: dict | None = None,
) -> AggResult:
    """Aggregate each HGT parameter group over the clients whose group is active
    (non-zero), so rare relations are not diluted by structurally-zero deltas."""
    if not aggregator.coordinatewise:
        raise ValueError(
            f"relation_aware aggregation only supports coordinate-wise rules, not {aggregator.name!r}"
        )
    weights = weights.to(deltas.device, torch.float32)
    active = layout.active_groups(deltas).cpu().numpy()
    out = torch.zeros(deltas.shape[1], dtype=deltas.dtype, device=deltas.device)
    patterns: dict[tuple[bool, ...], list[str]] = {}
    for gi, group in enumerate(layout.group_names):
        patterns.setdefault(tuple(bool(v) for v in active[:, gi]), []).append(group)
    for pattern, groups in patterns.items():
        rows = [i for i, flag in enumerate(pattern) if flag]
        if not rows:
            continue
        cols = layout.group_coords(groups).to(deltas.device)
        rows_t = torch.as_tensor(rows, device=deltas.device)
        sub = deltas.index_select(0, rows_t).index_select(1, cols)
        out[cols] = aggregator.aggregate(sub, weights[rows_t], ctx).delta
    return AggResult(out, info={"relation_aware": True, "num_patterns": len(patterns)})
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PY -m pytest tests/federated/test_aggregators.py -q`
Expected: `17 passed`

- [ ] **Step 5: Commit**

```bash
git add src/graphslm_ids/offline/federated/aggregators.py tests/federated/test_aggregators.py
git commit -m "feat(fl): add robust aggregators, clipping and relation-aware aggregation"
```

---

### Task 8: Delta-only screeners (`screening.py`)

**Files:**
- Create: `src/graphslm_ids/offline/federated/screening.py`
- Test: `tests/federated/test_screening.py`

- [ ] **Step 1: Write the failing tests**

```python
import pytest
import torch

from graphslm_ids.offline.federated.screening import (
    BrayCurtis,
    MultiSignal,
    NoScreen,
    TemporalAudit,
    build_screener,
)

IDS = list(range(10))


def _round(seed=0, n=10, d=300, noise=0.3):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(d, generator=g)
    return base + noise * torch.randn(n, d, generator=g)


def _drifting_rounds(rounds, seed=0, n=10, d=300):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(d, generator=g)
    for _ in range(rounds):
        base = base + 0.3 * torch.randn(d, generator=g)
        yield base + 0.3 * torch.randn(n, d, generator=g)


@pytest.mark.parametrize(
    "make",
    [lambda: MultiSignal(), lambda: TemporalAudit(warmup_rounds=1), lambda: BrayCurtis(variant="signed")],
    ids=["multi_signal", "temporal_audit", "bray_curtis_signed"],
)
def test_screeners_reject_a_scaled_outlier(make):
    x = _round()
    x[0] = 50.0 * x[0]
    result = make().screen(x, IDS, 1)
    assert result.decision[0] == "reject" and result.trust[0] == 0.0


def test_multi_signal_rejects_sign_flip_and_remembers_it():
    screener = MultiSignal()
    x = _round()
    x[0] = -x[0]
    assert screener.screen(x, IDS, 1).decision[0] == "reject"
    assert screener.trust_hist[0] < 0.5
    second = screener.screen(_round(seed=5), IDS, 2)
    assert second.features["trust"][0] < 0.85


def test_bray_curtis_abs_is_blind_to_pure_sign_flip_but_signed_is_not():
    x = _round()
    x[0] = -x[1]
    blind = BrayCurtis(variant="abs").screen(x, IDS, 1)
    assert blind.features["bc_score"][0] == pytest.approx(blind.features["bc_score"][1], rel=1e-6)
    assert BrayCurtis(variant="signed").screen(x, IDS, 1).decision[0] == "reject"


def test_bray_curtis_reputation_bans_repeat_offender():
    screener = BrayCurtis(variant="signed", penalty=0.6)
    for r in (1, 2):
        x = _round(seed=r)
        x[0] = -5.0 * x[0]
        screener.screen(x, IDS, r)
    result = screener.screen(_round(seed=3), IDS, 3)
    assert 0 in screener.banned and result.decision[0] == "reject"


def test_temporal_audit_has_low_false_rejection_on_benign_rounds():
    screener = TemporalAudit(warmup_rounds=2)
    rejected = 0
    for r, x in enumerate(_drifting_rounds(8), start=1):
        rejected += screener.screen(x, IDS, r).decision.count("reject")
    assert rejected / 80 <= 0.1


def test_temporal_audit_catches_client_that_turns_malicious():
    screener = TemporalAudit(warmup_rounds=2)
    decisions = []
    for r, x in enumerate(_drifting_rounds(6), start=1):
        if r >= 5:
            x[0] = -x[0]
        decisions.append(screener.screen(x, IDS, r).decision[0])
    assert decisions[4] == "reject" and decisions[5] == "reject"


def test_noscreen_and_builder():
    result = NoScreen().screen(torch.randn(3, 4), [0, 1, 2], 1)
    assert result.decision == ["accept"] * 3 and result.trust.tolist() == [1.0, 1.0, 1.0]
    assert isinstance(build_screener({"name": "temporal_audit"}), TemporalAudit)
    assert build_screener({"name": "bray_curtis", "params": {"variant": "signed"}}).variant == "signed"
    with pytest.raises(ValueError):
        build_screener({"name": "bogus"})
    with pytest.raises(ValueError):
        build_screener({"name": "bray_curtis", "params": {"variant": "bogus"}})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PY -m pytest tests/federated/test_screening.py -q`
Expected: FAIL — `ModuleNotFoundError: ... federated.screening`

- [ ] **Step 3: Write the implementation**

```python
"""Per-client update screening from deltas only (the server never sees data).

Each screener returns a trust in [0, 1] and a decision per client; the server
drops ``reject`` rows and aggregates the rest with weight ``trust * num_samples``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from graphslm_ids.offline.federated.linalg import (
    coordinate_median,
    cosine_to,
    mean_excluding_self,
    median_excluding_self,
    pairwise_cosine,
    robust_center_scale,
    robust_z,
)
from graphslm_ids.offline.federated.params import ParamLayout

ACCEPT, DOWNWEIGHT, REJECT = "accept", "downweight", "reject"


@dataclass
class ScreenResult:
    trust: np.ndarray
    decision: list[str]
    features: dict[str, np.ndarray] = field(default_factory=dict)


def reference_direction(deltas: torch.Tensor, prev_aggregate: torch.Tensor | None) -> torch.Tensor:
    """Previous round's aggregate (SignGuard-Sim convention); coordinate median at round 1."""
    if prev_aggregate is not None and float(prev_aggregate.norm()) > 0:
        return prev_aggregate.to(deltas.device, deltas.dtype)
    return coordinate_median(deltas)


def sign_agreement(deltas: torch.Tensor, ref: torch.Tensor) -> np.ndarray:
    ref_sign = torch.sign(ref)
    mask = ref_sign != 0
    denom = max(int(mask.sum()), 1)
    agree = (torch.sign(deltas[:, mask]) == ref_sign[mask]).sum(dim=1)
    return (agree.to(torch.float64) / denom).cpu().numpy()


def relation_deviation(deltas: torch.Tensor, layout: ParamLayout | None) -> np.ndarray:
    """max over active HGT relation groups of |log(group norm / median group norm)|."""
    n = deltas.shape[0]
    out = np.zeros(n)
    if layout is None:
        return out
    rel = [i for i, g in enumerate(layout.group_names) if g.startswith("rel:")]
    if not rel:
        return out
    norms = layout.group_norms(deltas)[:, rel].to(torch.float64).cpu().numpy()
    for j in range(norms.shape[1]):
        col = norms[:, j]
        act = col > 0
        if act.sum() < 2:
            continue
        med = float(np.median(col[act]))
        if med <= 0:
            continue
        out[act] = np.maximum(out[act], np.abs(np.log(col[act] / med)))
    return out


class Screener:
    name = "base"

    def screen(
        self,
        deltas: torch.Tensor,
        client_ids: list[int],
        round_idx: int,
        *,
        layout: ParamLayout | None = None,
        prev_aggregate: torch.Tensor | None = None,
    ) -> ScreenResult:
        raise NotImplementedError


class NoScreen(Screener):
    name = "none"

    def screen(self, deltas, client_ids, round_idx, *, layout=None, prev_aggregate=None):
        n = deltas.shape[0]
        return ScreenResult(np.ones(n), [ACCEPT] * n)


class MultiSignal(Screener):
    """Compact OptiGradTrust-style trust (Karami et al., Sci Rep 2026) without the
    VAE / Shapley / DDQN parts, which need server data. Per-signal suspicion has a 1
    robust-sigma dead zone and is combined by noisy-OR, so one saturated signal is
    enough to reject (an equal-weight mean only downweights a x50 update)."""

    name = "multi_signal"

    def __init__(
        self,
        z_max: float = 3.0,
        beta: float = 0.8,
        reject_below: float = 0.5,
        accept_above: float = 0.8,
        use_relation_groups: bool = True,
        scale_floor: float = 1e-3,
    ) -> None:
        if z_max <= 1.0:
            raise ValueError("z_max must be > 1")
        self.z_max = float(z_max)
        self.beta = float(beta)
        self.reject_below = float(reject_below)
        self.accept_above = float(accept_above)
        self.use_relation_groups = bool(use_relation_groups)
        self.scale_floor = float(scale_floor)
        self.trust_hist: dict[int, float] = {}

    def _low(self, z: np.ndarray) -> np.ndarray:
        return np.clip((-z - 1.0) / (self.z_max - 1.0), 0.0, 1.0)

    def _high(self, z: np.ndarray) -> np.ndarray:
        return np.clip((z - 1.0) / (self.z_max - 1.0), 0.0, 1.0)

    def screen(self, deltas, client_ids, round_idx, *, layout=None, prev_aggregate=None):
        ids = [int(c) for c in client_ids]
        ref = reference_direction(deltas, prev_aggregate)
        floor = self.scale_floor
        feats = {
            "cos_ref": cosine_to(deltas, ref).cpu().numpy(),
            "cos_others": mean_excluding_self(pairwise_cosine(deltas).cpu().numpy()),
            "log_norm": np.log(deltas.norm(dim=1).to(torch.float64).clamp_min(1e-30).cpu().numpy()),
            "sign_agree": sign_agreement(deltas, ref),
        }
        suspicion = [
            self._low(robust_z(feats["cos_ref"], floor)),
            self._low(robust_z(feats["cos_others"], floor)),
            self._high(np.abs(robust_z(feats["log_norm"], floor))),
            self._low(robust_z(feats["sign_agree"], floor)),
        ]
        if self.use_relation_groups and layout is not None:
            feats["relation_dev"] = relation_deviation(deltas, layout)
            suspicion.append(self._high(robust_z(feats["relation_dev"], floor)))
        trust_now = np.prod(1.0 - np.stack(suspicion), axis=0)
        trust = np.array(
            [
                self.beta * t + (1.0 - self.beta) * self.trust_hist[c] if c in self.trust_hist else t
                for c, t in zip(ids, trust_now)
            ]
        )
        decisions: list[str] = []
        weights = np.zeros(len(ids))
        for i, t in enumerate(trust):
            if t < self.reject_below:
                decisions.append(REJECT)
            elif t < self.accept_above:
                decisions.append(DOWNWEIGHT)
                weights[i] = t
            else:
                decisions.append(ACCEPT)
                weights[i] = 1.0
        for c, t in zip(ids, trust):
            self.trust_hist[c] = float(t)
        feats["trust"] = trust
        return ScreenResult(weights, decisions, feats)


class TemporalAudit(Screener):
    """History-aware Mahalanobis auditing inspired by SecureDyn-FL (Soomro et al.).

    Unsupervised reconstruction: the paper publishes neither its feature set nor its
    thresholds / forgetting factor, and trains its auditor on labelled malicious
    gradients. Every constant here is our choice (design spec §3.7)."""

    name = "temporal_audit"
    FEATURES = ("log_norm_ratio", "cos_ref", "sign_agree", "median_cos_others", "relation_dev")

    def __init__(
        self,
        alpha: float = 0.7,
        k_md: float = 3.0,
        k_dmd: float = 3.0,
        norm_ratio: float = 3.0,
        downweight: float = 0.5,
        warmup_rounds: int = 2,
        sketch_dim: int = 4096,
        sketch_decay: float = 0.5,
        seed: int = 0,
        use_relation_groups: bool = True,
        scale_floor: float = 1e-3,
    ) -> None:
        self.alpha = float(alpha)
        self.k_md = float(k_md)
        self.k_dmd = float(k_dmd)
        self.norm_ratio = float(norm_ratio)
        self.downweight = float(downweight)
        self.warmup_rounds = int(warmup_rounds)
        self.sketch_dim = int(sketch_dim)
        self.sketch_decay = float(sketch_decay)
        self.seed = int(seed)
        self.use_relation_groups = bool(use_relation_groups)
        self.scale_floor = float(scale_floor)
        self.center: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.md_center: float | None = None
        self.md_scale: float | None = None
        self.norm_ref: float | None = None
        self.prev_md: dict[int, float] = {}
        self.dmd_hist: list[float] = []
        self.sketch_hist: dict[int, torch.Tensor] = {}
        self._bucket: torch.Tensor | None = None
        self._sign: torch.Tensor | None = None

    def _ema(self, old, new):
        return new if old is None else self.alpha * old + (1.0 - self.alpha) * new

    def _sketch(self, deltas: torch.Tensor) -> torch.Tensor:
        d = deltas.shape[1]
        if self._bucket is None or self._bucket.numel() != d:
            gen = torch.Generator().manual_seed(self.seed)
            self._bucket = torch.randint(0, self.sketch_dim, (d,), generator=gen)
            self._sign = torch.randint(0, 2, (d,), generator=gen).to(torch.float32) * 2.0 - 1.0
        signed = deltas * self._sign.to(deltas.device, deltas.dtype)
        out = torch.zeros((deltas.shape[0], self.sketch_dim), dtype=deltas.dtype, device=deltas.device)
        out.index_add_(1, self._bucket.to(deltas.device), signed)
        return out.cpu()

    def screen(self, deltas, client_ids, round_idx, *, layout=None, prev_aggregate=None):
        ids = [int(c) for c in client_ids]
        n = len(ids)
        norms = deltas.norm(dim=1).to(torch.float64).cpu().numpy()
        med_norm = max(float(np.median(norms)), 1e-30)
        ref = reference_direction(deltas, prev_aggregate)
        feats = {
            "log_norm_ratio": np.log(np.maximum(norms, 1e-30) / med_norm),
            "cos_ref": cosine_to(deltas, ref).cpu().numpy(),
            "sign_agree": sign_agreement(deltas, ref),
            "median_cos_others": median_excluding_self(pairwise_cosine(deltas).cpu().numpy()),
        }
        if self.use_relation_groups and layout is not None:
            feats["relation_dev"] = relation_deviation(deltas, layout)
        x = np.stack([feats[k] for k in self.FEATURES if k in feats], axis=1)
        c_now, s_now = robust_center_scale(x, axis=0, floor=self.scale_floor)
        if self.center is not None and self.center.shape == c_now.shape:
            center, scale = self._ema(self.center, c_now), self._ema(self.scale, s_now)
        else:
            center, scale = c_now, s_now
        md = np.sqrt((((x - center) / scale) ** 2).sum(axis=1))
        mc_now, ms_now = robust_center_scale(md, floor=1e-6)
        mc_now = float(mc_now)
        ms_now = max(float(ms_now), 0.1 * mc_now, 1e-6)
        md_c = float(self._ema(self.md_center, mc_now))
        md_s = float(self._ema(self.md_scale, ms_now))
        ratio_md = (md - md_c) / (self.k_md * md_s)
        norm_ref = med_norm if self.norm_ref is None else self.norm_ref
        ratio_norm = norms / max(self.norm_ratio * norm_ref, 1e-30)
        ratio_dmd = np.full(n, np.nan)
        ratio_hist = np.full(n, np.nan)
        sketches = self._sketch(deltas)
        if round_idx > self.warmup_rounds:
            if len(self.dmd_hist) >= 3:
                hc, hs = robust_center_scale(np.asarray(self.dmd_hist), floor=1e-6)
                thr = max(float(hc) + self.k_dmd * max(float(hs), 0.1 * md_s), 1e-12)
                for i, c in enumerate(ids):
                    if c in self.prev_md:
                        ratio_dmd[i] = abs(md[i] - self.prev_md[c]) / thr
            hist_cos = np.full(n, np.nan)
            for i, c in enumerate(ids):
                h = self.sketch_hist.get(c)
                if h is not None:
                    hist_cos[i] = float(torch.nn.functional.cosine_similarity(sketches[i], h, dim=0))
            have = ~np.isnan(hist_cos)
            if have.sum() >= 3:
                ratio_hist[have] = -robust_z(hist_cos[have], floor=self.scale_floor) / self.k_md
        ratios = np.stack([ratio_md, ratio_norm, ratio_dmd, ratio_hist], axis=1)
        filled = np.where(np.isnan(ratios), -np.inf, ratios)
        n_exceed = (filled >= 1.0).sum(axis=1)
        worst = filled.max(axis=1)
        decisions: list[str] = []
        trust = np.zeros(n)
        for i in range(n):
            if n_exceed[i] == 0:
                decisions.append(ACCEPT)
                trust[i] = 1.0
            elif n_exceed[i] == 1 and worst[i] < 2.0:
                decisions.append(DOWNWEIGHT)
                trust[i] = self.downweight
            else:
                decisions.append(REJECT)
        self._update(ids, decisions, x, md, norms, sketches)
        feats.update(
            {"md": md, "ratio_md": ratio_md, "ratio_norm": ratio_norm, "ratio_dmd": ratio_dmd, "ratio_hist": ratio_hist}
        )
        return ScreenResult(trust, decisions, feats)

    def _update(self, ids, decisions, x, md, norms, sketches) -> None:
        acc = np.array([d == ACCEPT for d in decisions], dtype=bool)
        if acc.sum() >= 2 or self.center is None:
            rows = acc if acc.sum() >= 2 else np.ones(len(ids), dtype=bool)
            c_acc, s_acc = robust_center_scale(x[rows], axis=0, floor=self.scale_floor)
            mc, ms = robust_center_scale(md[rows], floor=1e-6)
            mc = float(mc)
            ms = max(float(ms), 0.1 * mc, 1e-6)
            if self.center is not None and self.center.shape != c_acc.shape:
                self.center = self.scale = None
            self.center = self._ema(self.center, c_acc)
            self.scale = self._ema(self.scale, s_acc)
            self.md_center = float(self._ema(self.md_center, mc))
            self.md_scale = float(self._ema(self.md_scale, ms))
            self.norm_ref = float(self._ema(self.norm_ref, float(np.median(norms[rows]))))
        for i, c in enumerate(ids):
            if acc[i] and c in self.prev_md:
                self.dmd_hist.append(abs(float(md[i]) - self.prev_md[c]))
            self.prev_md[c] = float(md[i])
            if acc[i]:
                h = self.sketch_hist.get(c)
                self.sketch_hist[c] = (
                    sketches[i].clone()
                    if h is None
                    else self.sketch_decay * h + (1.0 - self.sketch_decay) * sketches[i]
                )
        self.dmd_hist = self.dmd_hist[-1000:]


class BrayCurtis(Screener):
    """BPFLH (Zhu et al., TDSC 2026) Eq. 1/16/17 without the CKKS layer.

    ``variant='abs'`` follows the paper (|g|, hence blind to a pure sign flip);
    ``'signed'`` is our extension on [g+, g-] (ablation)."""

    name = "bray_curtis"

    def __init__(
        self,
        variant: str = "abs",
        m: float = 0.5,
        reputation_init: float = 1.0,
        penalty: float = 0.25,
        chunk: int = 1 << 20,
    ) -> None:
        if variant not in {"abs", "signed"}:
            raise ValueError(f"unknown bray_curtis variant {variant!r}; expected abs or signed")
        self.variant = variant
        self.m = float(m)
        self.reputation_init = float(reputation_init)
        self.penalty = float(penalty)
        self.chunk = int(chunk)
        self.reputation: dict[int, float] = {}
        self.banned: set[int] = set()

    def bc_matrix(self, deltas: torch.Tensor) -> np.ndarray:
        n = deltas.shape[0]
        num = torch.zeros((n, n), dtype=torch.float64, device=deltas.device)
        l1 = torch.zeros(n, dtype=torch.float64, device=deltas.device)
        for start in range(0, deltas.shape[1], self.chunk):
            c = deltas[:, start:start + self.chunk]
            if self.variant == "abs":
                a = c.abs()
                num += torch.cdist(a, a, p=1).to(torch.float64)
            else:
                pos, neg = c.clamp(min=0), (-c).clamp(min=0)
                num += (torch.cdist(pos, pos, p=1) + torch.cdist(neg, neg, p=1)).to(torch.float64)
            l1 += c.abs().sum(dim=1).to(torch.float64)
        bc = (num / (l1[:, None] + l1[None, :]).clamp_min(1e-30)).cpu().numpy()
        np.fill_diagonal(bc, 0.0)
        return bc

    def screen(self, deltas, client_ids, round_idx, *, layout=None, prev_aggregate=None):
        ids = [int(c) for c in client_ids]
        n = len(ids)
        bc = self.bc_matrix(deltas)
        score = bc.sum(axis=1) / max(n - 1, 1)
        theta = float(np.median(score) + self.m * np.std(score))
        flagged = score > theta + 1e-12
        decisions: list[str] = []
        trust = np.zeros(n)
        for i, c in enumerate(ids):
            self.reputation.setdefault(c, self.reputation_init)
            if c in self.banned:
                decisions.append(REJECT)
                continue
            if flagged[i]:
                self.reputation[c] -= self.penalty
                if self.reputation[c] < 0:
                    self.banned.add(c)
                decisions.append(REJECT)
            else:
                decisions.append(ACCEPT)
                trust[i] = 1.0
        feats = {
            "bc_score": score,
            "theta": np.full(n, theta),
            "reputation": np.array([self.reputation[c] for c in ids]),
        }
        return ScreenResult(trust, decisions, feats)


_SCREENERS: dict[str, type[Screener]] = {
    "none": NoScreen,
    "temporal_audit": TemporalAudit,
    "multi_signal": MultiSignal,
    "bray_curtis": BrayCurtis,
}


def build_screener(cfg: dict, *, seed: int = 0) -> Screener:
    name = str(cfg.get("name", "none")).lower()
    if name not in _SCREENERS:
        raise ValueError(f"unknown screener {name!r}; choose from {sorted(_SCREENERS)}")
    params = dict(cfg.get("params") or {})
    if name == "temporal_audit":
        params.setdefault("seed", int(seed))
    try:
        return _SCREENERS[name](**params)
    except TypeError as exc:
        raise ValueError(f"bad params for screener {name!r}: {exc}") from exc
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PY -m pytest tests/federated/test_screening.py -q`
Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add src/graphslm_ids/offline/federated/screening.py tests/federated/test_screening.py
git commit -m "feat(fl): add TemporalAudit, MultiSignal and Bray-Curtis update screeners"
```

---

### Task 9: FL server (`server.py`)

**Files:**
- Create: `src/graphslm_ids/offline/federated/server.py`
- Test: `tests/federated/test_server.py`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest
import torch

from graphslm_ids.offline.federated.aggregators import FedAvg, MultiKrum, TrimmedMean
from graphslm_ids.offline.federated.screening import NoScreen, ScreenResult, Screener
from graphslm_ids.offline.federated.server import ClientMessage, FLServer


class _HalfTrust(Screener):
    name = "half"

    def screen(self, deltas, client_ids, round_idx, *, layout=None, prev_aggregate=None):
        return ScreenResult(np.array([1.0, 0.0]), ["accept", "reject"])


def test_server_applies_weighted_fedavg_and_advances_round():
    server = FLServer(global_vec=torch.zeros(3), aggregator=FedAvg(), screener=NoScreen())
    report = server.step(
        [ClientMessage(0, 1, torch.tensor([1.0, 0.0, 0.0]), 1), ClientMessage(1, 1, torch.tensor([0.0, 2.0, 0.0]), 3)]
    )
    assert report.applied and server.round_idx == 2
    assert torch.allclose(server.global_vec, torch.tensor([0.25, 1.5, 0.0]))
    assert torch.allclose(server.prev_aggregate, torch.tensor([0.25, 1.5, 0.0]))


def test_server_rejects_invalid_messages():
    server = FLServer(global_vec=torch.zeros(2), aggregator=FedAvg(), screener=NoScreen())
    report = server.step(
        [
            ClientMessage(0, 1, torch.tensor([float("nan"), 0.0]), 1),
            ClientMessage(1, 5, torch.ones(2), 1),
            ClientMessage(2, 1, torch.ones(3), 1),
            ClientMessage(3, 1, torch.ones(2), 1),
        ]
    )
    decisions = report.decision_map()
    assert decisions == {0: "invalid", 1: "invalid", 2: "invalid", 3: "accept"}
    assert torch.allclose(server.global_vec, torch.ones(2))


def test_server_keeps_global_when_nothing_is_valid():
    server = FLServer(global_vec=torch.ones(2), aggregator=FedAvg(), screener=NoScreen())
    report = server.step([ClientMessage(0, 1, torch.tensor([float("inf"), 0.0]), 1)])
    assert not report.applied and server.round_idx == 2
    assert torch.allclose(server.global_vec, torch.ones(2))


def test_server_uses_screen_decisions_and_server_lr():
    server = FLServer(global_vec=torch.zeros(2), aggregator=FedAvg(), screener=_HalfTrust(), server_lr=0.5)
    report = server.step([ClientMessage(0, 1, torch.tensor([2.0, 2.0]), 1), ClientMessage(1, 1, torch.tensor([-9.0, -9.0]), 1)])
    assert report.decision_map() == {0: "accept", 1: "reject"}
    assert torch.allclose(server.global_vec, torch.tensor([1.0, 1.0]))


def test_server_marks_aggregator_dropped_clients_rejected():
    g = torch.Generator().manual_seed(0)
    base = torch.randn(20, generator=g)
    deltas = base + 0.1 * torch.randn(10, 20, generator=g)
    deltas[0] = -50.0 * deltas[1]
    server = FLServer(global_vec=torch.zeros(20), aggregator=MultiKrum(f_ratio=0.2), screener=NoScreen())
    report = server.step([ClientMessage(i, 1, deltas[i], 1) for i in range(10)])
    assert report.decision_map()[0] == "reject"


def test_server_validates_configuration():
    with pytest.raises(ValueError):
        FLServer(global_vec=torch.zeros(2), aggregator=FedAvg(), screener=NoScreen(), weighting="bogus")
    with pytest.raises(ValueError):
        FLServer(global_vec=torch.zeros(2), aggregator=FedAvg(), screener=NoScreen(), relation_aware=True)
    from graphslm_ids.offline.federated.params import ParamLayout

    layout = ParamLayout.from_state_dict({"classifier.0.weight": torch.zeros(2)})
    with pytest.raises(ValueError):
        FLServer(global_vec=torch.zeros(2), aggregator=MultiKrum(), screener=NoScreen(), layout=layout, relation_aware=True)
    server = FLServer(global_vec=torch.zeros(2), aggregator=TrimmedMean(trim_k=0), screener=NoScreen(), layout=layout, relation_aware=True)
    server.step([ClientMessage(0, 1, torch.ones(2), 1)])
    assert torch.allclose(server.global_vec, torch.ones(2))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PY -m pytest tests/federated/test_server.py -q`
Expected: FAIL — `ModuleNotFoundError: ... federated.server`

- [ ] **Step 3: Write the implementation**

```python
"""FL coordinator: integrity checks -> clipping -> screening -> robust aggregation.

The server only ever receives ``ClientMessage`` (round id, delta, sample count).
It never learns which clients are malicious; the simulation grades its decisions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from graphslm_ids.offline.federated.aggregators import Aggregator, clip_deltas, relation_aware_aggregate
from graphslm_ids.offline.federated.params import ParamLayout
from graphslm_ids.offline.federated.records import to_jsonable
from graphslm_ids.offline.federated.screening import REJECT, Screener

INVALID = "invalid"


@dataclass
class ClientMessage:
    client_id: int
    round_idx: int
    delta: torch.Tensor
    num_samples: int


@dataclass
class ClientDecision:
    client_id: int
    decision: str
    trust: float
    weight: float
    norm: float | None
    reason: str = ""


@dataclass
class RoundReport:
    round_idx: int
    decisions: list[ClientDecision]
    applied: bool
    aggregate_norm: float
    seconds: float
    screen_features: dict[str, Any] = field(default_factory=dict)
    agg_info: dict[str, Any] = field(default_factory=dict)

    def decision_map(self) -> dict[int, str]:
        return {d.client_id: d.decision for d in self.decisions}


class FLServer:
    def __init__(
        self,
        *,
        global_vec: torch.Tensor,
        aggregator: Aggregator,
        screener: Screener,
        layout: ParamLayout | None = None,
        clip: dict | None = None,
        relation_aware: bool = False,
        weighting: str = "samples",
        server_lr: float = 1.0,
        device: str | torch.device = "cpu",
    ) -> None:
        if weighting not in {"samples", "uniform"}:
            raise ValueError(f"unknown weighting {weighting!r}; expected samples or uniform")
        if relation_aware and layout is None:
            raise ValueError("relation_aware aggregation needs a ParamLayout")
        if relation_aware and not aggregator.coordinatewise:
            raise ValueError(f"relation_aware does not support aggregator {aggregator.name!r}")
        self.device = torch.device(device)
        self.global_vec = global_vec.detach().to(self.device, torch.float32).clone()
        self.aggregator = aggregator
        self.screener = screener
        self.layout = layout
        self.clip = dict(clip or {"mode": "none"})
        self.relation_aware = bool(relation_aware)
        self.weighting = weighting
        self.server_lr = float(server_lr)
        self.round_idx = 1
        self.prev_aggregate: torch.Tensor | None = None

    @property
    def dim(self) -> int:
        return int(self.global_vec.numel())

    def validate(self, msg: ClientMessage) -> str | None:
        if int(msg.round_idx) != self.round_idx:
            return f"stale_round:{msg.round_idx}"
        if msg.delta.ndim != 1 or int(msg.delta.numel()) != self.dim:
            return f"shape:{tuple(msg.delta.shape)}"
        if not bool(torch.isfinite(msg.delta).all()):
            return "non_finite"
        return None

    def step(self, messages: list[ClientMessage]) -> RoundReport:
        start = time.perf_counter()
        r = self.round_idx
        decisions: list[ClientDecision] = []
        valid: list[ClientMessage] = []
        for msg in messages:
            reason = self.validate(msg)
            if reason:
                decisions.append(ClientDecision(int(msg.client_id), INVALID, 0.0, 0.0, None, reason))
            else:
                valid.append(msg)
        if not valid:
            self.round_idx += 1
            return RoundReport(r, decisions, False, 0.0, time.perf_counter() - start)

        ids = [int(m.client_id) for m in valid]
        deltas = torch.stack([m.delta.detach().to(self.device, torch.float32) for m in valid])
        deltas, norms = clip_deltas(
            deltas,
            mode=str(self.clip.get("mode", "none")),
            factor=float(self.clip.get("factor", 1.0)),
            value=self.clip.get("value"),
        )
        screen = self.screener.screen(deltas, ids, r, layout=self.layout, prev_aggregate=self.prev_aggregate)
        if self.weighting == "samples":
            base = torch.tensor([float(m.num_samples) for m in valid])
        else:
            base = torch.ones(len(valid))
        weights = base * torch.as_tensor(screen.trust, dtype=torch.float32)
        keep = [i for i, d in enumerate(screen.decision) if d != REJECT and float(weights[i]) > 0.0]
        final = list(screen.decision)
        applied, agg_norm, agg_info = False, 0.0, {}
        if keep:
            idx = torch.as_tensor(keep, device=self.device)
            sub = deltas.index_select(0, idx)
            sub_w = weights[keep]
            ctx = {"prev_aggregate": self.prev_aggregate}
            if self.relation_aware:
                result = relation_aware_aggregate(self.aggregator, sub, sub_w, self.layout, ctx)
            else:
                result = self.aggregator.aggregate(sub, sub_w, ctx)
            if result.selected is not None:
                for j, chosen in enumerate(result.selected.cpu().tolist()):
                    if not chosen:
                        final[keep[j]] = REJECT
            delta = result.delta.to(self.device)
            self.global_vec += self.server_lr * delta
            self.prev_aggregate = delta.detach().clone()
            applied, agg_norm, agg_info = True, float(delta.norm()), to_jsonable(result.info)
        kept = set(keep)
        for i, cid in enumerate(ids):
            rejected = final[i] == REJECT or i not in kept
            decisions.append(
                ClientDecision(
                    cid,
                    REJECT if rejected else final[i],
                    float(screen.trust[i]),
                    0.0 if rejected else float(weights[i]),
                    float(norms[i]),
                )
            )
        self.round_idx += 1
        return RoundReport(
            r,
            decisions,
            applied,
            agg_norm,
            time.perf_counter() - start,
            to_jsonable(screen.features),
            agg_info,
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PY -m pytest tests/federated/test_server.py -q`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/graphslm_ids/offline/federated/server.py tests/federated/test_server.py
git commit -m "feat(fl): add FL server with integrity checks, screening and aggregation"
```

---

### Task 10: Metrics (`metrics.py`)

**Files:**
- Create: `src/graphslm_ids/offline/federated/metrics.py`
- Test: `tests/federated/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from graphslm_ids.offline.federated.metrics import (
    compact_metrics,
    convergence_round,
    detection_counts,
    detection_rates,
    merge_counts,
    metrics_from_confusion,
)

NAMES = {0: "Benign", 1: "XSS", 2: "DDoS-ICMP_Flood"}


def _f1(p, r):
    return 2 * p * r / (p + r)


def test_metrics_from_confusion_rates_and_families():
    cm = np.array([[8, 2, 0], [1, 3, 0], [0, 0, 5]])
    m = metrics_from_confusion(cm, NAMES, families={"web": ["XSS"], "ddos": ["DDoS-ICMP_Flood"], "none": ["Nope"]})
    assert m["count"] == 19
    assert m["benign_false_alarm_rate"] == pytest.approx(0.2)
    assert m["attack_detection_rate"] == pytest.approx(1 - 1 / 9)
    assert m["alarm_precision"] == pytest.approx(1 - 2 / 10)
    assert m["family_recall"]["web"] == pytest.approx(0.75)
    assert m["family_recall"]["none"] is None
    expected = np.mean([_f1(8 / 9, 0.8), _f1(3 / 5, 0.75), 1.0])
    assert m["macro_f1"] == pytest.approx(expected)
    assert m["per_class"]["XSS"]["support"] == 4


def test_macro_f1_ignores_classes_without_support():
    cm = np.array([[5, 0, 0], [0, 5, 0], [0, 0, 0]])
    assert metrics_from_confusion(cm, NAMES)["macro_f1"] == pytest.approx(1.0)


def test_detection_counts_and_rates():
    counts = detection_counts({0: "reject", 1: "accept", 2: "downweight", 3: "invalid", 4: "accept"}, {0, 4})
    assert counts["tp"] == 1 and counts["fn"] == 1 and counts["fp"] == 1 and counts["tn"] == 2
    assert counts["downweighted_benign"] == 1 and counts["invalid"] == 1
    total = merge_counts({"tp": 1}, counts)
    rates = detection_rates(total)
    assert rates["tpr"] == pytest.approx(2 / 3)
    assert rates["fpr"] == pytest.approx(1 / 3)
    assert rates["malicious_acceptance_rate"] == pytest.approx(1 / 3)
    assert detection_rates({})["tpr"] is None


def test_convergence_round_and_compact():
    assert convergence_round([(1, 0.1), (2, 0.5), (3, 0.97), (4, 1.0)]) == 3
    assert convergence_round([(1, None)]) is None
    full = {"macro_f1": 0.5, "per_class": {}, "confusion": [[1]], "clean": {"macro_f1": 0.4, "per_class": {}}}
    assert compact_metrics(full) == {"macro_f1": 0.5, "clean": {"macro_f1": 0.4}}
    assert compact_metrics(None) == {}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PY -m pytest tests/federated/test_metrics.py -q`
Expected: FAIL — `ModuleNotFoundError: ... federated.metrics`

- [ ] **Step 3: Write the implementation**

```python
"""IDS and poisoning metrics for FL runs.

The eval loop builds confusion matrices directly (noisy labels and, when given, the
clean answer key). Macro-F1 averages classes with support > 0, exactly like the
centralized trainer, so numbers are comparable."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from graphslm_ids.offline.training.train_hgt_flow_classifier import to_torch_batch

FLAGGED = {"reject", "invalid"}
COMPACT_KEYS = (
    "count",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "loss",
    "benign_false_alarm_rate",
    "attack_detection_rate",
    "alarm_precision",
    "family_recall",
)


def _confusion(y: torch.Tensor, pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    idx = y.long().reshape(-1).cpu() * num_classes + pred.long().reshape(-1).cpu()
    return torch.bincount(idx, minlength=num_classes * num_classes).view(num_classes, num_classes)


def confusion_from_loader(
    model: torch.nn.Module,
    loader,
    edge_types: list[tuple[str, str, str]],
    device: torch.device,
    use_semantic_edge_weights: bool,
    num_classes: int,
    clean_labels: torch.Tensor | None = None,
) -> dict[str, Any]:
    model.eval()
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    cm_clean = torch.zeros_like(cm) if clean_labels is not None else None
    loss_sum, count = 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            nf, ei, ew, sm, sl = to_torch_batch(batch, edge_types, device, use_semantic_edge_weights)
            if sl.numel() == 0:
                continue
            logits = model(nf, ei, edge_weight_dict=ew)[sm].float()
            loss_sum += float(F.cross_entropy(logits, sl, reduction="sum"))
            count += int(sl.numel())
            pred = logits.argmax(dim=1)
            cm += _confusion(sl, pred, num_classes)
            if cm_clean is not None:
                gids = torch.as_tensor(np.asarray(batch.seed_flow_ids, dtype=np.int64))
                cm_clean += _confusion(clean_labels[gids], pred, num_classes)
    return {
        "cm": cm.numpy(),
        "cm_clean": None if cm_clean is None else cm_clean.numpy(),
        "loss": loss_sum / max(count, 1),
        "count": count,
    }


def _safe_div(num: float, den: float) -> float | None:
    return float(num) / float(den) if den else None


def metrics_from_confusion(
    cm: np.ndarray,
    label_names: dict[int, str],
    families: dict[str, list[str]] | None = None,
    benign: str = "Benign",
) -> dict[str, Any]:
    cm = np.asarray(cm, dtype=np.int64)
    c = cm.shape[0]
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    precision = np.divide(tp, predicted, out=np.zeros(c), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(c), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2.0 * precision * recall, denom, out=np.zeros(c), where=denom > 0)
    present = support > 0
    total = float(cm.sum())
    out: dict[str, Any] = {
        "count": int(total),
        "accuracy": float(tp.sum() / total) if total else None,
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "weighted_f1": float((f1 * support).sum() / support.sum()) if support.sum() else 0.0,
        "per_class": {
            label_names.get(i, str(i)): {
                "support": int(support[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
            }
            for i in range(c)
        },
    }
    name_to_id = {name: idx for idx, name in label_names.items()}
    b = name_to_id.get(benign)
    if b is not None and b < c:
        attack = [i for i in range(c) if i != b]
        out["benign_false_alarm_rate"] = _safe_div(support[b] - cm[b, b], support[b])
        missed = _safe_div(cm[attack, b].sum(), support[attack].sum())
        out["attack_detection_rate"] = None if missed is None else 1.0 - missed
        false_alarm_share = _safe_div(cm[b, attack].sum(), cm[:, attack].sum())
        out["alarm_precision"] = None if false_alarm_share is None else 1.0 - false_alarm_share
    if families:
        family_recall: dict[str, float | None] = {}
        for fam, members in families.items():
            ids = [name_to_id[m] for m in members if m in name_to_id and name_to_id[m] < c]
            family_recall[fam] = _safe_div(tp[ids].sum(), support[ids].sum()) if ids else None
        out["family_recall"] = family_recall
    return out


def detection_counts(decisions: dict[int, str], malicious: set[int]) -> dict[str, int]:
    counts = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
        "downweighted_benign": 0,
        "downweighted_malicious": 0,
        "invalid": 0,
    }
    for cid, decision in decisions.items():
        bad = int(cid) in malicious
        flagged = decision in FLAGGED
        if decision == "invalid":
            counts["invalid"] += 1
        if decision == "downweight":
            counts["downweighted_malicious" if bad else "downweighted_benign"] += 1
        if bad:
            counts["tp" if flagged else "fn"] += 1
        else:
            counts["fp" if flagged else "tn"] += 1
    return counts


def merge_counts(total: dict[str, int], new: dict[str, int]) -> dict[str, int]:
    return {k: int(total.get(k, 0)) + int(new.get(k, 0)) for k in sorted(set(total) | set(new))}


def detection_rates(counts: dict[str, int]) -> dict[str, float | None]:
    tp, fp, fn, tn = (int(counts.get(k, 0)) for k in ("tp", "fp", "fn", "tn"))
    tpr = tp / (tp + fn) if tp + fn else None
    return {
        "tpr": tpr,
        "fpr": fp / (fp + tn) if fp + tn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "malicious_acceptance_rate": None if tpr is None else 1.0 - tpr,
    }


def convergence_round(history: list[tuple[int, float | None]], frac: float = 0.95) -> int | None:
    values = [(int(r), float(s)) for r, s in history if s is not None]
    if not values:
        return None
    best = max(s for _r, s in values)
    target = frac * best if best > 0 else best
    for r, s in values:
        if s >= target:
            return r
    return None


def compact_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not metrics:
        return {}
    out = {k: metrics[k] for k in COMPACT_KEYS if k in metrics}
    if metrics.get("clean"):
        out["clean"] = {k: metrics["clean"][k] for k in COMPACT_KEYS if k in metrics["clean"]}
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PY -m pytest tests/federated/test_metrics.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/graphslm_ids/offline/federated/metrics.py tests/federated/test_metrics.py
git commit -m "feat(fl): add confusion-matrix eval and poisoning detection metrics"
```

---

### Task 11: Client local trainer (`local_trainer.py`)

**Files:**
- Create: `src/graphslm_ids/offline/federated/local_trainer.py`
- Test: `tests/federated/test_local_trainer.py`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest
import torch

from graphslm_ids.models.hgt import HeteroGraphTransformer
from graphslm_ids.offline.federated.client_graph import build_client_artifact
from graphslm_ids.offline.federated.local_trainer import (
    ClientState,
    LocalTrainer,
    local_class_weights,
    round_lr,
)
from graphslm_ids.offline.federated.params import ParamLayout
from graphslm_ids.offline.training.neighbor_sampling import HeteroNeighborSampler, InMemoryNeighborBackend
from graphslm_ids.offline.training.noise_consensus import EACSController

CLIENT_CFG = {
    "local_epochs": 1,
    "batch_seed_flows": 4,
    "weight_decay": 0.0,
    "grad_clip_norm": 1.0,
    "optimizer_state": "reset",
    "loss_type": "focal",
    "focal_gamma": 1.5,
    "label_smoothing": 0.05,
    "stratified_batch_sampling": True,
    "drop_edge_prob": 0.0,
    "num_workers": 0,
    "gcl": {"enabled": False},
}
EVIDENCE = "rel:packet__evidence_injection__technique"


def _setup(art, flow_ids, cfg=None, gcl_map=None):
    flow_ids = np.asarray(flow_ids, dtype=np.int64)
    backend = InMemoryNeighborBackend(build_client_artifact(art, flow_ids))
    torch.manual_seed(0)
    model = HeteroGraphTransformer(
        node_input_dims=dict(backend.feature_dims),
        edge_types=list(backend.edge_types),
        num_classes=3,
        num_tactics=backend.num_tactics,
        hidden_dim=8,
        num_layers=2,
        num_heads=2,
        node_types=list(backend.feature_dims) + ["tactic"],
    )
    layout = ParamLayout.from_model(model)
    labels = np.asarray(art.flow_y)[flow_ids]
    state = ClientState(
        client_id=0,
        flow_ids=flow_ids,
        flow_labels=labels,
        backend=backend,
        sampler=HeteroNeighborSampler(backend, hops=2, seed=0),
        class_weights=local_class_weights(labels, 3),
    )
    trainer = LocalTrainer(model, layout, backend.edge_types, "cpu", cfg or CLIENT_CFG, gcl_map=gcl_map)
    return trainer, layout, state


def test_local_class_weights_ignore_absent_classes():
    w = local_class_weights(np.array([0, 0, 0, 1]), 3, method="cb", beta=0.9, cap=10.0)
    assert w[2].item() == 1.0
    assert float(w[:2].mean()) == pytest.approx(1.0)
    assert float(w[1]) > float(w[0])
    with pytest.raises(ValueError):
        local_class_weights(np.array([0]), 2, method="bogus")


def test_round_lr_schedules():
    cfg = {"lr": 1e-2, "lr_min": 1e-4, "lr_schedule": "cosine_rounds"}
    assert round_lr(cfg, 1, 10) == pytest.approx(1e-2)
    assert round_lr(cfg, 10, 10) < round_lr(cfg, 5, 10) < round_lr(cfg, 1, 10)
    assert round_lr({"lr": 0.1, "lr_schedule": "constant"}, 7, 10) == 0.1
    with pytest.raises(ValueError):
        round_lr({"lr": 0.1, "lr_schedule": "bogus"}, 1, 2)


def test_local_trainer_returns_finite_delta(tiny_artifact):
    trainer, layout, state = _setup(tiny_artifact, np.arange(30))
    start = layout.flatten(trainer.model.state_dict())
    update = trainer.train(state, start, round_idx=1, lr=1e-2, seed=0)
    assert update.delta.shape == (layout.numel,)
    assert bool(torch.isfinite(update.delta).all())
    assert update.steps > 0 and float(update.delta.abs().sum()) > 0
    assert update.num_samples == 30 and state.rounds_trained == 1


def test_relation_absent_from_client_has_exactly_zero_delta(tiny_artifact):
    y = np.asarray(tiny_artifact.flow_y)
    trainer, layout, state = _setup(tiny_artifact, np.flatnonzero(y == 0))
    update = trainer.train(state, layout.flatten(trainer.model.state_dict()), 1, 1e-2, 0)
    assert int(torch.count_nonzero(update.delta[layout.group_coords([EVIDENCE])])) == 0
    trainer2, layout2, state2 = _setup(tiny_artifact, np.flatnonzero(y == 1))
    update2 = trainer2.train(state2, layout2.flatten(trainer2.model.state_dict()), 1, 1e-2, 0)
    assert int(torch.count_nonzero(update2.delta[layout2.group_coords([EVIDENCE])])) > 0


def test_eacs_soft_target_path_runs_after_warmup(tiny_artifact):
    trainer, layout, state = _setup(tiny_artifact, np.arange(30))
    y = torch.as_tensor(np.asarray(tiny_artifact.flow_y))
    state.eacs = EACSController(
        suspect_mask=y == 1,
        anchor_mask=torch.zeros_like(y, dtype=torch.bool),
        benign_class_id=0,
        num_classes=3,
        warmup_epochs=1,
    )
    update = trainer.train(state, layout.flatten(trainer.model.state_dict()), 3, 1e-2, 0)
    assert bool(torch.isfinite(update.delta).all())
    assert state.eacs.epoch_stats()["suspects_seen"] > 0


def test_gcl_path_runs(tiny_artifact):
    cfg = {**CLIENT_CFG, "gcl": {"enabled": True, "weight": 0.3, "temperature": 0.1, "n_negatives": 2}}
    trainer, layout, state = _setup(tiny_artifact, np.arange(30), cfg=cfg, gcl_map={1: [(0, 1.0)], 2: [(2, 1.0)]})
    assert trainer.gcl_enabled
    update = trainer.train(state, layout.flatten(trainer.model.state_dict()), 1, 1e-2, 0)
    assert bool(torch.isfinite(update.delta).all())


def test_persisted_optimizer_state_lives_on_cpu(tiny_artifact):
    trainer, layout, state = _setup(tiny_artifact, np.arange(30), cfg={**CLIENT_CFG, "optimizer_state": "persist"})
    vec = layout.flatten(trainer.model.state_dict())
    update = trainer.train(state, vec, 1, 1e-2, 0)
    assert state.optimizer_state is not None
    for per_param in state.optimizer_state["state"].values():
        for value in per_param.values():
            if isinstance(value, torch.Tensor):
                assert value.device.type == "cpu"
    trainer.train(state, vec + update.delta, 2, 1e-2, 1)
    assert state.rounds_trained == 2


def test_trainer_rejects_bad_config(tiny_artifact):
    with pytest.raises(ValueError):
        _setup(tiny_artifact, np.arange(5), cfg={**CLIENT_CFG, "optimizer_state": "bogus"})
    with pytest.raises(ValueError):
        _setup(tiny_artifact, np.arange(5), cfg={**CLIENT_CFG, "loss_type": "bogus"})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PY -m pytest tests/federated/test_local_trainer.py -q`
Expected: FAIL — `ModuleNotFoundError: ... federated.local_trainer`

- [ ] **Step 3: Write the implementation**

```python
"""Client-side local HGT training for one FL round.

Lean recipe (approved design): focal/CE with class-balanced weights from the
client's LOCAL labels, optional GCL auxiliary loss and EACS soft targets. No HGAA,
TEW, DDP, EMA or OOM retry. A relation that never appears in the client's batches
gets grad None, AdamW skips it, and its delta is exactly zero.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from graphslm_ids.offline.federated.params import ParamLayout
from graphslm_ids.offline.training.noise_consensus import neighbor_consensus, soft_target_focal_loss
from graphslm_ids.offline.training.train_hgt_flow_classifier import (
    _compute_train_loss,
    _drop_edges,
    gcl_auxiliary_loss,
    make_neighbor_loader,
    to_torch_batch,
)

BURST_KEYS = (("flow", "burst_neighbor", "flow"), ("flow", "rev_burst_neighbor", "flow"))
CONTAINS = ("flow", "contains", "packet")
LOSS_TYPES = ("ce", "focal", "cb_focal")


def local_class_weights(
    labels: np.ndarray, num_classes: int, method: str = "cb", beta: float = 0.985, cap: float = 4.0
) -> torch.Tensor:
    """Class weights from one client's labels. Absent classes get 1.0 and stay out of
    the mean-1 normalisation (the centralized helper lets them skew it)."""
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float64)
    present = counts > 0
    w = np.ones(num_classes, dtype=np.float64)
    if method == "cb":
        w[present] = (1.0 - beta) / (1.0 - np.power(beta, counts[present]))
    elif method == "inverse":
        w[present] = counts[present].sum() / (present.sum() * counts[present])
    else:
        raise ValueError(f"unknown class_weight_method {method!r}; expected cb or inverse")
    if present.any():
        w[present] /= w[present].mean()
    return torch.from_numpy(np.minimum(w, float(cap)).astype(np.float32))


def round_lr(client_cfg: dict[str, Any], round_idx: int, total_rounds: int) -> float:
    lr = float(client_cfg.get("lr", 1e-3))
    schedule = str(client_cfg.get("lr_schedule", "cosine_rounds")).lower()
    if schedule == "constant":
        return lr
    if schedule == "cosine_rounds":
        lr_min = float(client_cfg.get("lr_min", 0.0))
        progress = (int(round_idx) - 1) / max(int(total_rounds), 1)
        return lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"unknown lr_schedule {schedule!r}; expected cosine_rounds or constant")


@dataclass
class ClientState:
    client_id: int
    flow_ids: np.ndarray
    flow_labels: np.ndarray
    backend: Any
    sampler: Any
    class_weights: torch.Tensor | None = None
    eacs: Any = None
    optimizer_state: dict | None = None
    rounds_trained: int = 0


@dataclass
class LocalUpdate:
    client_id: int
    delta: torch.Tensor
    num_samples: int
    steps: int
    seconds: float
    metrics: dict[str, Any]


def _to_cpu(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu").clone()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    return obj


class LocalTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        layout: ParamLayout,
        edge_types: list[tuple[str, str, str]],
        device: str | torch.device,
        client_cfg: dict[str, Any],
        *,
        use_semantic_edge_weights: bool = True,
        gcl_map: dict[int, list[tuple[int, float]]] | None = None,
    ) -> None:
        self.model = model
        self.layout = layout
        self.edge_types = list(edge_types)
        self.device = torch.device(device)
        self.cfg = dict(client_cfg)
        if str(self.cfg.get("optimizer_state", "reset")) not in {"reset", "persist"}:
            raise ValueError("client.optimizer_state must be 'reset' or 'persist'")
        self.loss_type = str(self.cfg.get("loss_type", "focal")).lower()
        if self.loss_type not in LOSS_TYPES:
            raise ValueError(f"unknown loss_type {self.loss_type!r}; expected one of {LOSS_TYPES}")
        self.label_smoothing = float(self.cfg.get("label_smoothing", 0.0))
        self.focal_gamma = float(self.cfg.get("focal_gamma", 2.0))
        gcl = self.cfg.get("gcl") or {}
        self.gcl_map = dict(gcl_map or {})
        self.gcl_enabled = bool(gcl.get("enabled", False)) and bool(self.gcl_map)
        self.gcl_weight = float(gcl.get("weight", 0.3))
        self.gcl_temperature = float(gcl.get("temperature", 0.1))
        self.gcl_n_negatives = int(gcl.get("n_negatives", 16))
        self.use_sew = bool(use_semantic_edge_weights)

    def _loader(self, state: ClientState, seed: int):
        loader_cfg = {
            "train": {
                "batch_seed_flows": int(self.cfg.get("batch_seed_flows", 256)),
                "stratified_batch_sampling": bool(self.cfg.get("stratified_batch_sampling", True)),
            },
            "dataloader": {
                "num_workers": int(self.cfg.get("num_workers", 0)),
                "pin_memory": bool(self.cfg.get("pin_memory", False)),
                "persistent_workers": False,
                "prefetch_factor": 2,
            },
        }
        loader, _ = make_neighbor_loader(
            state.flow_ids, state.sampler, loader_cfg, shuffle=True, seed=seed, class_labels=state.flow_labels
        )
        return loader

    def _consensus(self, logits, ei, sm, sl):
        parts = [ei[k] for k in BURST_KEYS if k in ei and ei[k].numel() > 0]
        if not parts:
            return None
        seed_idx = torch.nonzero(sm, as_tuple=False).reshape(-1)
        return neighbor_consensus(torch.softmax(logits.detach().float(), dim=1), torch.cat(parts, dim=1), seed_idx, sl)

    def _gcl_loss(self, batch, nf, ei, sm, sl, x_dict):
        contains = ei.get(CONTAINS)
        if contains is None or contains.numel() == 0:
            return None
        flow_local, pkt_local = contains[0], contains[1]
        keep = sm[flow_local]
        if not bool(keep.any()):
            return None
        dense = torch.full((int(nf["flow"].shape[0]),), -1, dtype=torch.long, device=self.device)
        dense[torch.nonzero(sm, as_tuple=False).reshape(-1)] = sl
        pkt_class = dense[flow_local[keep]]
        valid = pkt_class >= 0
        if not bool(valid.any()):
            return None
        tech_lg = batch.local_to_global.get("technique")
        if tech_lg is None:
            return None
        tech_ids = tech_lg.detach().cpu().tolist() if isinstance(tech_lg, torch.Tensor) else np.asarray(tech_lg).tolist()
        g2l = {int(g): i for i, g in enumerate(tech_ids)}
        local_map: dict[int, list[tuple[int, float]]] = {}
        for cls_idx, pairs in self.gcl_map.items():
            remapped = [(g2l[int(t)], float(w)) for t, w in pairs if int(t) in g2l]
            if remapped:
                local_map[int(cls_idx)] = remapped
        if not local_map:
            return None
        return gcl_auxiliary_loss(
            node_embeddings={"packet": x_dict["packet"], "technique": x_dict["technique"]},
            seed_packet_ids=pkt_local[keep][valid],
            seed_packet_flow_label_idx=pkt_class[valid],
            class_to_technique_idx=local_map,
            num_techniques=int(x_dict["technique"].shape[0]),
            temperature=self.gcl_temperature,
            n_negatives=self.gcl_n_negatives,
            device=self.device,
        )

    def _step(self, batch, state: ClientState, round_idx: int):
        nf, ei, ew, sm, sl = to_torch_batch(batch, self.edge_types, self.device, self.use_sew)
        if sl.numel() == 0:
            return None
        drop = float(self.cfg.get("drop_edge_prob", 0.0) or 0.0)
        if drop > 0.0:
            ei, ew = _drop_edges(ei, ew, drop)
        if self.gcl_enabled:
            x_dict = self.model.encode(nf, ei, edge_weight_dict=ew)
            logits = self.model.classifier(x_dict["flow"])
        else:
            x_dict = None
            logits = self.model(nf, ei, edge_weight_dict=ew)
        seed_logits = logits[sm].float()
        weight = None if state.class_weights is None else state.class_weights.to(self.device)
        if state.eacs is not None:
            gids = torch.as_tensor(np.asarray(batch.seed_flow_ids, dtype=np.int64), device=self.device)
            consensus = self._consensus(logits, ei, sm, sl) if state.eacs.lambda_disambig < 1.0 else None
            targets = state.eacs.soft_targets(seed_logits, gids, sl, epoch=round_idx, consensus=consensus)
            loss = soft_target_focal_loss(
                seed_logits,
                targets,
                weight,
                loss_type=self.loss_type,
                label_smoothing=self.label_smoothing,
                focal_gamma=self.focal_gamma,
            )
        else:
            loss = _compute_train_loss(
                seed_logits,
                sl,
                weight,
                loss_type=self.loss_type,
                label_smoothing=self.label_smoothing,
                focal_gamma=self.focal_gamma,
            )
        if x_dict is not None:
            aux = self._gcl_loss(batch, nf, ei, sm, sl, x_dict)
            if aux is not None:
                loss = loss + self.gcl_weight * aux
        return loss, seed_logits.detach(), sl

    def train(
        self, state: ClientState, global_vec: torch.Tensor, round_idx: int, lr: float, seed: int
    ) -> LocalUpdate:
        start = time.perf_counter()
        torch.manual_seed(int(seed))
        state.sampler.rng = np.random.default_rng(int(seed))
        self.model.load_state_dict(self.layout.unflatten(global_vec))
        self.model.train()
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=float(lr),
            weight_decay=float(self.cfg.get("weight_decay", 0.0)),
            eps=float(self.cfg.get("adamw_eps", 1e-6)),
            betas=tuple(self.cfg.get("adamw_betas", (0.9, 0.95))),
        )
        persist = str(self.cfg.get("optimizer_state", "reset")) == "persist"
        if persist and state.optimizer_state is not None:
            optimizer.load_state_dict(state.optimizer_state)
            for group in optimizer.param_groups:
                group["lr"] = float(lr)
        max_steps = self.cfg.get("max_local_steps")
        max_steps = int(max_steps) if max_steps else None
        clip = float(self.cfg.get("grad_clip_norm", 0.0) or 0.0)
        steps = examples = correct = skipped = 0
        loss_sum = 0.0
        done = False
        for epoch in range(int(self.cfg.get("local_epochs", 1))):
            for batch in self._loader(state, int(seed) + epoch):
                out = self._step(batch, state, round_idx)
                if out is None:
                    skipped += 1
                    continue
                loss, seed_logits, sl = out
                if not bool(torch.isfinite(loss)):
                    skipped += 1
                    continue
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, clip)
                optimizer.step()
                n = int(sl.numel())
                steps += 1
                examples += n
                loss_sum += float(loss.detach()) * n
                correct += int((seed_logits.argmax(dim=1) == sl).sum())
                if max_steps is not None and steps >= max_steps:
                    done = True
                    break
            if done:
                break
        if persist:
            state.optimizer_state = _to_cpu(optimizer.state_dict())
        state.rounds_trained += 1
        trained = self.layout.flatten(self.model.state_dict())
        delta = trained - global_vec.detach().to("cpu", torch.float32)
        return LocalUpdate(
            client_id=state.client_id,
            delta=delta,
            num_samples=int(state.flow_ids.shape[0]),
            steps=steps,
            seconds=time.perf_counter() - start,
            metrics={
                "loss": loss_sum / max(examples, 1),
                "accuracy": correct / max(examples, 1),
                "examples": examples,
                "skipped_batches": skipped,
            },
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PY -m pytest tests/federated/test_local_trainer.py -q`
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add src/graphslm_ids/offline/federated/local_trainer.py tests/federated/test_local_trainer.py
git commit -m "feat(fl): add lean client-side HGT trainer with EACS and GCL"
```

---

### Task 12: Experiment driver (`simulation.py`)

**Files:**
- Modify: `tests/federated/conftest.py` (append the `smoke_cfg` fixture)
- Create: `src/graphslm_ids/offline/federated/simulation.py`
- Test: `tests/federated/test_simulation.py`

- [ ] **Step 1: Add the smoke-config fixture and write the failing tests**

Append to `tests/federated/conftest.py`:

```python
@pytest.fixture
def smoke_cfg(tiny_paths, tmp_path):
    """Factory: a CPU smoke config on the tiny artifact writing to tmp_path/<name>."""
    from graphslm_ids.offline.federated.config import apply_overrides, load_fl_config

    def make(name: str = "run", **overrides):
        base = {
            "data.graph_npz": str(tiny_paths["graph_npz"]),
            "data.graph_meta_json": str(tiny_paths["graph_meta_json"]),
            "data.splits_json": str(tiny_paths["splits_json"]),
            "data.clean_eval_labels": str(tiny_paths["clean_eval_labels"]),
            "data.packet_dtype": "float32",
            "model": {"hidden_dim": 8, "num_layers": 2, "num_heads": 2, "dropout": 0.0, "ffn_multiplier": 2},
            "sampler": {
                "hops": 2,
                "fanouts": {},
                "reverse_fanouts": {},
                "always_include_all_techniques": True,
                "always_include_all_tactics": True,
            },
            "partition": {
                "scheme": "iid",
                "num_clients": 3,
                "alpha": 0.5,
                "min_flows_per_client": 1,
                "max_flows_per_client": None,
                "seed": 0,
            },
            "client.batch_seed_flows": 8,
            "client.gcl.enabled": True,
            "client.gcl.class_technique_map_csv": str(tiny_paths["class_technique_map_csv"]),
            "client.eacs.enabled": True,
            "client.eacs.anchor_mask_npy": str(tiny_paths["anchor_mask_npy"]),
            "client.eacs.warmup_rounds": 1,
            "client.eacs.suspect_classes": ["XSS"],
            "federated.rounds": 2,
            "federated.seed": 0,
            "eval.every": 1,
            "eval.monitor": "val_macro_f1_clean",
            "eval.batch_seed_flows": 16,
            "output.dir": str(tmp_path / name),
            "device": "cpu",
        }
        base.update(overrides)
        return apply_overrides(load_fl_config(None), base)

    return make
```

`tests/federated/test_simulation.py`:

```python
import json

import pytest
import torch

from graphslm_ids.models.hgt import HeteroGraphTransformer
from graphslm_ids.offline.federated.config import apply_overrides
from graphslm_ids.offline.federated.simulation import (
    load_context,
    run_experiment,
    select_malicious,
    select_participants,
)


def _records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_selection_helpers_are_deterministic():
    assert select_participants(10, 1.0, 0, 3) == list(range(10))
    half = select_participants(10, 0.5, 0, 3)
    assert half == select_participants(10, 0.5, 0, 3) and len(half) == 5
    assert select_malicious(10, 0.2, 0) == select_malicious(10, 0.2, 0)
    assert len(select_malicious(10, 0.2, 0)) == 2 and select_malicious(10, 0.0, 0) == []


def test_fl_smoke_sign_flip_trimmed_mean(smoke_cfg, tmp_path):
    cfg = smoke_cfg(
        "fl",
        **{
            "attack.name": "sign_flip",
            "attack.malicious_ratio": 0.34,
            "defense.aggregator": {"name": "trimmed_mean", "params": {}},
            "defense.assumed_malicious_ratio": 0.34,
        },
    )
    summary = run_experiment(cfg)
    out = tmp_path / "fl"
    for name in ("config.resolved.yaml", "partition.json", "rounds.jsonl", "fl_summary.json", "global_final.pt", "global_best.pt"):
        assert (out / name).exists(), name
    assert len(summary["malicious_clients"]) == 1
    records = _records(out / "rounds.jsonl")
    assert [r["round"] for r in records] == [1, 2]
    assert records[0]["attack_active"] and len(records[0]["decisions"]) == 3
    assert 0.0 <= summary["final"]["test"]["macro_f1"] <= 1.0
    assert "clean" in summary["final"]["test"]
    assert summary["detection"]["tp"] + summary["detection"]["fn"] == 2
    ckpt = torch.load(out / "global_final.pt", map_location="cpu", weights_only=False)
    model = HeteroGraphTransformer(
        node_input_dims=ckpt["node_input_dims"],
        edge_types=[tuple(e) for e in ckpt["edge_types"]],
        num_classes=ckpt["num_classes"],
        num_tactics=ckpt["num_tactics"],
        hidden_dim=8,
        num_layers=2,
        num_heads=2,
        node_types=list(ckpt["node_input_dims"]) + ["tactic"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    assert ckpt["label_names"][1] == "XSS" and ckpt["flow_feature_stats"]["mean"]


def test_fl_smoke_omniscient_alie_skips_attacker_training(smoke_cfg, tmp_path):
    cfg = smoke_cfg(
        "alie",
        **{
            "attack.name": "alie",
            "attack.malicious_ratio": 0.34,
            "defense.screen": {"name": "temporal_audit", "params": {"warmup_rounds": 1}},
        },
    )
    summary = run_experiment(cfg)
    first = _records(tmp_path / "alie" / "rounds.jsonl")[0]
    assert str(summary["malicious_clients"][0]) not in first["client_metrics"]
    assert summary["screen"] == "temporal_audit"


def test_fl_smoke_relation_aware_multi_signal_min_max(smoke_cfg):
    cfg = smoke_cfg(
        "ra",
        **{
            "attack.name": "min_max",
            "attack.malicious_ratio": 0.34,
            "defense.screen": {"name": "multi_signal", "params": {}},
            "defense.relation_aware": True,
            "defense.clip": {"mode": "median", "factor": 1.0},
        },
    )
    summary = run_experiment(cfg)
    assert summary["aggregator"] == "fedavg" and summary["rounds"] == 2


def test_centralized_mode_trains_on_union_of_client_flows(smoke_cfg, tmp_path):
    summary = run_experiment(
        smoke_cfg("central", **{"federated.mode": "centralized", "attack.name": "sign_flip", "attack.malicious_ratio": 0.34})
    )
    assert summary["num_clients"] == 1 and summary["malicious_clients"] == [] and summary["attack"] == "none"
    part = json.loads((tmp_path / "central" / "partition.json").read_text(encoding="utf-8"))
    assert part["scheme"] == "centralized" and part["sizes"] == [32]


def test_local_only_mode_reports_every_client(smoke_cfg, tmp_path):
    summary = run_experiment(smoke_cfg("local", **{"federated.mode": "local_only"}))
    assert len(summary["per_client"]) == 3
    assert summary["aggregate"]["test_macro_f1"]["mean"] is not None
    assert (tmp_path / "local" / "fl_summary.json").exists()


def test_context_reuse_and_validation(smoke_cfg):
    cfg = smoke_cfg("ctx")
    ctx = load_context(cfg)
    run_experiment(cfg, context=ctx)
    assert ctx.client_cache
    with pytest.raises(ValueError):
        run_experiment(apply_overrides(cfg, {"data.split_protocol": "temporal"}), context=ctx)
    with pytest.raises(ValueError):
        run_experiment(apply_overrides(cfg, {"federated.mode": "bogus"}))
    with pytest.raises(ValueError):
        run_experiment(apply_overrides(cfg, {"eval.monitor": "bogus"}))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PY -m pytest tests/federated/test_simulation.py -q`
Expected: FAIL — `ModuleNotFoundError: ... federated.simulation`

- [ ] **Step 3: Write the implementation**

```python
"""FL experiment driver: ``fl`` / ``centralized`` / ``local_only`` on one v3 artifact.

Round protocol (design spec §3.9): select participants -> local HGT training on each
client's own graph -> experiment-only Byzantine hook replaces attacker deltas ->
server integrity/clip/screen/aggregate -> grade decisions against ground truth ->
periodic validation of the global model on the full graph (experiment-only).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from graphslm_ids.models.hgt import HeteroGraphTransformer
from graphslm_ids.offline.federated.aggregators import build_aggregator
from graphslm_ids.offline.federated.attacks import Attack, AttackContext, build_attack
from graphslm_ids.offline.federated.client_graph import build_client_artifact, packet_owner
from graphslm_ids.offline.federated.config import apply_overrides
from graphslm_ids.offline.federated.local_trainer import (
    ClientState,
    LocalTrainer,
    LocalUpdate,
    local_class_weights,
    round_lr,
)
from graphslm_ids.offline.federated.metrics import (
    compact_metrics,
    confusion_from_loader,
    convergence_round,
    detection_counts,
    detection_rates,
    merge_counts,
    metrics_from_confusion,
)
from graphslm_ids.offline.federated.params import ParamLayout
from graphslm_ids.offline.federated.partition import ClientPartition, make_partition, subsample_partition
from graphslm_ids.offline.federated.records import append_jsonl, to_jsonable, write_json
from graphslm_ids.offline.federated.screening import build_screener
from graphslm_ids.offline.federated.server import ClientMessage, FLServer
from graphslm_ids.offline.training.hetero_graph_artifact import HeteroGraphArtifact, load_v3_artifact
from graphslm_ids.offline.training.neighbor_sampling import HeteroNeighborSampler, InMemoryNeighborBackend
from graphslm_ids.offline.training.noise_consensus import EACSController, build_eacs_controller
from graphslm_ids.offline.training.train_hgt_flow_classifier import (
    _set_tew_state,
    backend_splits,
    label_name_mapping,
    load_class_technique_map,
    make_neighbor_loader,
    resolve_device,
    set_seed,
)

VALID_MODES = ("fl", "centralized", "local_only")
VALID_MONITORS = ("val_macro_f1", "val_macro_f1_clean", "val_loss")


@dataclass
class ExperimentContext:
    data_key: str
    artifact: HeteroGraphArtifact
    backend: InMemoryNeighborBackend
    labels: np.ndarray
    train_ids: np.ndarray
    val_ids: np.ndarray
    test_ids: np.ndarray
    label_names: dict[int, str]
    num_classes: int
    clean_labels: torch.Tensor | None
    packet_owner: np.ndarray
    client_cache: dict[str, list[InMemoryNeighborBackend]] = field(default_factory=dict)


def data_key(cfg: dict[str, Any]) -> str:
    payload = json.dumps(to_jsonable(cfg["data"]), sort_keys=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def load_context(cfg: dict[str, Any]) -> ExperimentContext:
    data = cfg["data"]
    if str(data.get("artifact_version", "")).lower() != "v3":
        raise ValueError("the FL testbed requires data.artifact_version: v3")
    artifact = load_v3_artifact(
        Path(data["graph_npz"]),
        Path(data["graph_meta_json"]),
        add_reverse_edges=bool(data.get("add_reverse_edges", True)),
        packet_dtype=str(data.get("packet_dtype", "float32")),
    )
    backend = InMemoryNeighborBackend(artifact)
    labels = np.asarray(artifact.flow_y, dtype=np.int64)
    split_cfg = {"data": data, "train": {"val_ratio": 0.1, "test_ratio": 0.1}}
    train_ids, val_ids, test_ids = backend_splits(
        backend, labels, split_cfg, int(cfg["federated"].get("seed", 42))
    )
    label_names = label_name_mapping(backend.manifest, labels)
    num_classes = max(len(label_names), int(labels.max()) + 1)
    clean = None
    if data.get("clean_eval_labels"):
        arr = np.load(data["clean_eval_labels"])
        if arr.shape[0] != labels.shape[0]:
            raise ValueError(
                f"clean_eval_labels has {arr.shape[0]} rows but the artifact has {labels.shape[0]} flows"
            )
        clean = torch.as_tensor(np.asarray(arr, dtype=np.int64))
    return ExperimentContext(
        data_key=data_key(cfg),
        artifact=artifact,
        backend=backend,
        labels=labels,
        train_ids=np.asarray(train_ids, dtype=np.int64),
        val_ids=np.asarray(val_ids, dtype=np.int64),
        test_ids=np.asarray(test_ids, dtype=np.int64),
        label_names=label_names,
        num_classes=num_classes,
        clean_labels=clean,
        packet_owner=packet_owner(artifact),
    )


def select_participants(num_clients: int, fraction: float, seed: int, round_idx: int) -> list[int]:
    k = max(1, int(round(float(fraction) * num_clients)))
    if k >= num_clients:
        return list(range(num_clients))
    rng = np.random.default_rng(int(seed) * 100_003 + int(round_idx))
    return sorted(int(c) for c in rng.choice(num_clients, size=k, replace=False))


def select_malicious(num_clients: int, ratio: float, seed: int) -> list[int]:
    m = min(max(int(round(float(ratio) * num_clients)), 0), num_clients)
    if m == 0:
        return []
    rng = np.random.default_rng(int(seed) + 7919)
    return sorted(int(c) for c in rng.choice(num_clients, size=m, replace=False))


def build_model(cfg: dict[str, Any], backend: InMemoryNeighborBackend, num_classes: int) -> HeteroGraphTransformer:
    dims = {nt: int(d) for nt, d in backend.feature_dims.items()}
    m = cfg["model"]
    return HeteroGraphTransformer(
        node_input_dims=dims,
        edge_types=list(backend.edge_types),
        num_classes=int(num_classes),
        num_tactics=int(backend.num_tactics),
        hidden_dim=int(m["hidden_dim"]),
        num_layers=int(m["num_layers"]),
        num_heads=int(m["num_heads"]),
        dropout=float(m["dropout"]),
        ffn_multiplier=int(m["ffn_multiplier"]),
        node_types=list(dims) + ["tactic"],
    )


def make_sampler(
    cfg: dict[str, Any], backend: InMemoryNeighborBackend, flow_stats: dict[str, list[float]], seed: int
) -> HeteroNeighborSampler:
    s = cfg["sampler"]
    return HeteroNeighborSampler(
        backend,
        hops=int(s.get("hops") or cfg["model"]["num_layers"]),
        fanouts=dict(s.get("fanouts") or {}),
        reverse_fanouts=dict(s.get("reverse_fanouts") or {}),
        always_include_all_tactics=bool(s.get("always_include_all_tactics", True)),
        always_include_all_techniques=bool(s.get("always_include_all_techniques", True)),
        flow_feature_stats=flow_stats,
        standardize_flow_features=bool(cfg["data"].get("standardize_flow_features", True)),
        seed=int(seed),
    )


def federated_flow_stats(backend: InMemoryNeighborBackend, client_flow_ids: list[np.ndarray]) -> dict[str, list[float]]:
    """Round-0 protocol message: each client sends (n, sum x, sum x^2) of its flow_x."""
    dim = int(backend.feature_dims["flow"])
    n = 0
    total = np.zeros(dim, dtype=np.float64)
    total_sq = np.zeros(dim, dtype=np.float64)
    for ids in client_flow_ids:
        rows = backend.get_flow_features(np.asarray(ids, dtype=np.int64)).astype(np.float64)
        n += rows.shape[0]
        total += rows.sum(axis=0)
        total_sq += np.square(rows).sum(axis=0)
    mean = total / max(n, 1)
    std = np.maximum(np.sqrt(np.maximum(total_sq / max(n, 1) - np.square(mean), 0.0)), 1e-6)
    return {"mean": mean.tolist(), "std": std.tolist()}


def partition_key(partition: ClientPartition) -> str:
    digest = hashlib.sha1()
    for ids in partition.client_flow_ids:
        digest.update(np.asarray(ids, dtype=np.int64).tobytes())
        digest.update(b"|")
    return digest.hexdigest()


def client_backends(cfg: dict[str, Any], ctx: ExperimentContext, partition: ClientPartition) -> list[InMemoryNeighborBackend]:
    key = partition_key(partition)
    cached = ctx.client_cache.get(key)
    if cached is not None:
        return cached
    ctx.client_cache.clear()
    backends = [
        InMemoryNeighborBackend(build_client_artifact(ctx.artifact, ids, owner=ctx.packet_owner))
        for ids in partition.client_flow_ids
    ]
    if bool(cfg["federated"].get("cache_clients", True)):
        ctx.client_cache[key] = backends
    return backends


def build_base_eacs(cfg: dict[str, Any], ctx: ExperimentContext) -> EACSController | None:
    e = cfg["client"].get("eacs") or {}
    if not e.get("enabled"):
        return None
    anchor = np.load(e["anchor_mask_npy"]) if e.get("anchor_mask_npy") else None
    return build_eacs_controller(
        artifact=ctx.artifact,
        num_classes=ctx.num_classes,
        label_mapping={name: idx for idx, name in ctx.label_names.items()},
        suspect_classes=list(e.get("suspect_classes") or []),
        warmup_epochs=int(e.get("warmup_rounds", 5)),
        ema_decay=float(e.get("ema_decay", 0.9)),
        lambda_disambig=float(e.get("lambda_disambig", 0.7)),
        mitre_dir=str(e.get("mitre_dir", "data/mitre")),
        anchor_mask=anchor,
    )


def build_clients(
    cfg: dict[str, Any],
    ctx: ExperimentContext,
    partition: ClientPartition,
    flow_stats: dict[str, list[float]],
    base_eacs: EACSController | None,
) -> list[ClientState]:
    client_cfg = cfg["client"]
    seed = int(cfg["federated"].get("seed", 42))
    balanced = str(client_cfg.get("class_weight", "balanced")).lower() == "balanced"
    eacs_cfg = client_cfg.get("eacs") or {}
    states: list[ClientState] = []
    for cid, (ids, backend) in enumerate(zip(partition.client_flow_ids, client_backends(cfg, ctx, partition))):
        ids = np.asarray(ids, dtype=np.int64)
        labels = ctx.labels[ids]
        weights = (
            local_class_weights(
                labels,
                ctx.num_classes,
                method=str(client_cfg.get("class_weight_method", "cb")),
                beta=float(client_cfg.get("cb_beta", 0.985)),
                cap=float(client_cfg.get("class_weight_cap", 4.0)),
            )
            if balanced
            else None
        )
        eacs = None
        if base_eacs is not None:
            # Shared (read-only) masks, per-client EMA buffer: EACS state stays local.
            eacs = EACSController(
                suspect_mask=base_eacs.suspect_mask,
                anchor_mask=base_eacs.anchor_mask,
                benign_class_id=base_eacs.benign_class_id,
                num_classes=base_eacs.num_classes,
                warmup_epochs=base_eacs.warmup_epochs,
                ema_decay=float(eacs_cfg.get("ema_decay", 0.9)),
                lambda_disambig=base_eacs.lambda_disambig,
            )
        states.append(
            ClientState(
                client_id=cid,
                flow_ids=ids,
                flow_labels=labels,
                backend=backend,
                sampler=make_sampler(cfg, backend, flow_stats, seed + cid),
                class_weights=weights,
                eacs=eacs,
            )
        )
    return states


def build_gcl_map(cfg: dict[str, Any], backend: InMemoryNeighborBackend) -> dict[int, list[tuple[int, float]]]:
    g = cfg["client"].get("gcl") or {}
    if not g.get("enabled"):
        return {}
    manifest = backend.manifest or {}
    return load_class_technique_map(
        csv_path=Path(g.get("class_technique_map_csv", "data/mitre/class_technique_map.csv")),
        label_mapping={str(k): int(v) for k, v in (manifest.get("label_mapping") or {}).items()},
        num_techniques=int(manifest.get("num_techniques", backend.num_techniques)),
        technique_id_to_idx={str(k): int(v) for k, v in (manifest.get("technique_id_to_idx") or {}).items()},
    )


def stratified_subset(ids: np.ndarray, labels: np.ndarray, cap: int | None, seed: int) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64)
    if not cap or ids.shape[0] <= int(cap):
        return ids
    part = subsample_partition(ClientPartition([ids], "eval", None, seed), labels, int(cap), seed)
    return part.client_flow_ids[0]


class Evaluator:
    """Experiment-only grading of the global model on the full graph."""

    def __init__(self, cfg, ctx: ExperimentContext, model, flow_stats, device) -> None:
        self.ctx = ctx
        self.model = model
        self.device = device
        self.seed = int(cfg["federated"].get("seed", 42)) + 7
        self.sampler = make_sampler(cfg, ctx.backend, flow_stats, self.seed)
        self.use_sew = bool(cfg["data"].get("use_semantic_edge_weights", True))
        self.families = cfg["eval"].get("families") or {}
        self.loader_cfg = {
            "train": {"batch_seed_flows": int(cfg["eval"].get("batch_seed_flows", 512)), "stratified_batch_sampling": False},
            "dataloader": {
                "num_workers": int(cfg["client"].get("num_workers", 0)),
                "pin_memory": False,
                "persistent_workers": False,
                "prefetch_factor": 2,
            },
        }

    def evaluate(self, flow_ids: np.ndarray) -> dict[str, Any]:
        self.sampler.rng = np.random.default_rng(self.seed)
        loader, _ = make_neighbor_loader(
            np.asarray(flow_ids, dtype=np.int64), self.sampler, self.loader_cfg, shuffle=False, seed=self.seed
        )
        raw = confusion_from_loader(
            self.model,
            loader,
            list(self.ctx.backend.edge_types),
            self.device,
            self.use_sew,
            self.ctx.num_classes,
            self.ctx.clean_labels,
        )
        out = metrics_from_confusion(raw["cm"], self.ctx.label_names, self.families)
        out["loss"] = raw["loss"]
        out["confusion"] = raw["cm"].tolist()
        if raw["cm_clean"] is not None:
            clean = metrics_from_confusion(raw["cm_clean"], self.ctx.label_names, self.families)
            clean["confusion"] = raw["cm_clean"].tolist()
            out["clean"] = clean
        return out


def monitor_score(metrics: dict[str, Any], monitor: str) -> float | None:
    if monitor == "val_macro_f1":
        return metrics.get("macro_f1")
    if monitor == "val_macro_f1_clean":
        return (metrics.get("clean") or metrics).get("macro_f1")
    if monitor == "val_loss":
        loss = metrics.get("loss")
        return None if loss is None else -float(loss)
    raise ValueError(f"unknown eval.monitor {monitor!r}; expected one of {VALID_MONITORS}")


def _rss_gb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024**3
    except Exception:  # pragma: no cover - psutil missing or restricted
        return None


def save_checkpoint(path, layout, vec, cfg, ctx, flow_stats, round_idx, val_metrics, test_metrics, extra) -> None:
    torch.save(
        {
            "model_state_dict": layout.unflatten(vec),
            "config": to_jsonable(cfg),
            "node_input_dims": {k: int(v) for k, v in ctx.backend.feature_dims.items()},
            "edge_types": [list(e) for e in ctx.backend.edge_types],
            "num_classes": int(ctx.num_classes),
            "num_tactics": int(ctx.backend.num_tactics),
            "label_names": dict(ctx.label_names),
            "flow_feature_stats": flow_stats,
            "epoch": int(round_idx),
            "val_metrics": to_jsonable(compact_metrics(val_metrics)),
            "test_metrics": to_jsonable(compact_metrics(test_metrics)) if test_metrics else None,
            "fl": to_jsonable(extra),
        },
        path,
    )


def _craft(
    attack: Attack,
    knowledge: str,
    updates: dict[int, LocalUpdate],
    participants: list[int],
    mal_set: set[int],
    mal_now: list[int],
    dim: int,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    honest_rows = [updates[c].delta for c in mal_now if c in updates]
    benign_rows = [updates[c].delta for c in participants if c not in mal_set and c in updates]
    empty = torch.zeros((0, dim))
    ctx = AttackContext(
        honest=(torch.stack(honest_rows) if honest_rows else empty).to(device),
        benign=(torch.stack(benign_rows) if benign_rows else empty).to(device),
        num_participants=len(participants),
        knowledge=knowledge,
    )
    crafted = attack.craft(ctx, len(mal_now)).detach().cpu()
    return {c: crafted[j] for j, c in enumerate(mal_now)}


def _run_federated(*, cfg, ctx, out, clients, trainer, evaluator, layout, init_vec, val_ids, flow_stats, partition, device, mode):
    fed = cfg["federated"]
    rounds = int(fed["rounds"])
    seed = int(fed.get("seed", 42))
    attack_cfg = cfg["attack"]
    attack = build_attack(attack_cfg)
    knowledge = str(attack_cfg.get("knowledge", "omniscient"))
    start_round = int(attack_cfg.get("start_round", 1))
    malicious = [] if attack.name == "none" else select_malicious(len(clients), float(attack_cfg.get("malicious_ratio", 0.0)), seed)
    mal_set = set(malicious)
    defense = cfg["defense"]
    server = FLServer(
        global_vec=init_vec,
        aggregator=build_aggregator(
            defense.get("aggregator") or {},
            assumed_malicious_ratio=float(defense.get("assumed_malicious_ratio", 0.2)),
            seed=seed,
        ),
        screener=build_screener(defense.get("screen") or {}, seed=seed),
        layout=layout,
        clip=defense.get("clip"),
        relation_aware=bool(defense.get("relation_aware", False)),
        weighting=str(defense.get("weighting", "samples")),
        server_lr=float(fed.get("server_lr", 1.0)),
        device=device,
    )
    rounds_path = out / "rounds.jsonl"
    if rounds_path.exists():
        rounds_path.unlink()
    every = max(1, int(cfg["eval"].get("every", 1)))
    monitor = str(cfg["eval"]["monitor"])
    bytes_per_update = 4 * layout.numel
    totals: dict[str, int] = {}
    history: list[dict[str, Any]] = []
    best: dict[str, Any] = {"score": None, "round": 0, "vec": None, "val": None}
    last_val: dict[str, Any] | None = None
    cost: dict[str, Any] = {
        "client_train_seconds": 0.0,
        "server_seconds": 0.0,
        "eval_seconds": 0.0,
        "bytes_up": 0,
        "bytes_down": 0,
        "peak_rss_gb": None,
        "peak_vram_gb": None,
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    run_start = time.perf_counter()
    for r in range(1, rounds + 1):
        round_start = time.perf_counter()
        lr = round_lr(cfg["client"], r, rounds)
        participants = select_participants(len(clients), float(fed.get("participation", 1.0)), seed, r)
        attack_on = attack.name != "none" and r >= start_round
        need_honest = attack.needs_honest or knowledge == "partial"
        updates: dict[int, LocalUpdate] = {}
        t_train = time.perf_counter()
        for cid in participants:
            if attack_on and cid in mal_set and not need_honest:
                continue
            updates[cid] = trainer.train(clients[cid], server.global_vec, r, lr, seed=seed + 100_003 * r + 1_009 * cid)
        train_seconds = time.perf_counter() - t_train
        deltas = {cid: upd.delta for cid, upd in updates.items()}
        mal_now = [c for c in participants if c in mal_set]
        if attack_on and mal_now:
            deltas.update(_craft(attack, knowledge, updates, participants, mal_set, mal_now, layout.numel, device))
        messages = [ClientMessage(c, r, deltas[c], int(clients[c].flow_ids.shape[0])) for c in participants]
        report = server.step(messages)
        det = detection_counts(report.decision_map(), mal_set)
        totals = merge_counts(totals, det)
        cost["client_train_seconds"] += train_seconds
        cost["server_seconds"] += report.seconds
        cost["bytes_up"] += bytes_per_update * len(participants)
        cost["bytes_down"] += bytes_per_update * len(participants)
        record: dict[str, Any] = {
            "round": r,
            "lr": lr,
            "participants": participants,
            "malicious": mal_now,
            "attack_active": attack_on,
            "applied": report.applied,
            "aggregate_norm": report.aggregate_norm,
            "decisions": [asdict(d) for d in report.decisions],
            "screen_features": report.screen_features,
            "agg_info": report.agg_info,
            "detection": det,
            "client_metrics": {c: {**u.metrics, "steps": u.steps, "seconds": u.seconds} for c, u in updates.items()},
            "train_seconds": train_seconds,
            "server_seconds": report.seconds,
        }
        if r % every == 0 or r == rounds:
            t_eval = time.perf_counter()
            trainer.model.load_state_dict(layout.unflatten(server.global_vec))
            last_val = evaluator.evaluate(val_ids)
            score = monitor_score(last_val, monitor)
            history.append(
                {
                    "round": r,
                    "val_macro_f1": last_val.get("macro_f1"),
                    "val_macro_f1_clean": (last_val.get("clean") or {}).get("macro_f1"),
                    "val_loss": last_val.get("loss"),
                    "score": score,
                }
            )
            if score is not None and (best["score"] is None or score > best["score"]):
                best = {"score": score, "round": r, "vec": server.global_vec.detach().cpu().clone(), "val": last_val}
            record["val"] = compact_metrics(last_val)
            cost["eval_seconds"] += time.perf_counter() - t_eval
        rss = _rss_gb()
        if rss is not None:
            cost["peak_rss_gb"] = max(cost["peak_rss_gb"] or 0.0, rss)
        record["round_seconds"] = time.perf_counter() - round_start
        append_jsonl(rounds_path, record)
        val_msg = "" if "val" not in record else f" val_macro_f1={record['val'].get('macro_f1')}"
        print(f"[fl] round {r}/{rounds} applied={report.applied}{val_msg} ({record['round_seconds']:.1f}s)", flush=True)

    final_vec = server.global_vec.detach().cpu().clone()
    trainer.model.load_state_dict(layout.unflatten(final_vec))
    test_final = evaluator.evaluate(ctx.test_ids)
    if best["vec"] is None:
        best = {"score": None, "round": rounds, "vec": final_vec, "val": last_val}
    trainer.model.load_state_dict(layout.unflatten(best["vec"]))
    test_best = evaluator.evaluate(ctx.test_ids)
    if device.type == "cuda":
        cost["peak_vram_gb"] = torch.cuda.max_memory_allocated(device) / 1024**3
    cost["wall_seconds"] = time.perf_counter() - run_start
    cost["mean_round_seconds"] = cost["wall_seconds"] / max(rounds, 1)
    extra = {
        "mode": mode,
        "malicious_clients": malicious,
        "attack": attack.name,
        "aggregator": server.aggregator.name,
        "screen": server.screener.name,
    }
    if cfg["output"].get("save_checkpoints", True):
        save_checkpoint(out / "global_final.pt", layout, final_vec, cfg, ctx, flow_stats, rounds, last_val, test_final, extra)
        save_checkpoint(out / "global_best.pt", layout, best["vec"], cfg, ctx, flow_stats, best["round"], best["val"], test_best, extra)
    return {
        "experiment": cfg["experiment"].get("name"),
        "labels": cfg["experiment"].get("labels") or {},
        "mode": mode,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "num_clients": len(clients),
        "client_sizes": partition.sizes(),
        "malicious_clients": malicious,
        "attack": attack.name,
        "aggregator": server.aggregator.name,
        "screen": server.screener.name,
        "num_params": layout.numel,
        "rounds": rounds,
        "monitor": monitor,
        "best_round": best["round"],
        "best_score": best["score"],
        "final": {"val": last_val, "test": test_final},
        "best": {"val": best["val"], "test": test_best},
        "detection": {**totals, **detection_rates(totals)},
        "convergence_round": convergence_round([(h["round"], h["score"]) for h in history]),
        "history": history,
        "cost": cost,
        "config": cfg,
    }


def _run_local_only(*, cfg, ctx, out, clients, trainer, evaluator, layout, init_vec, val_ids, flow_stats, partition, device, mode):
    rounds = int(cfg["federated"]["rounds"])
    seed = int(cfg["federated"].get("seed", 42))
    rounds_path = out / "rounds.jsonl"
    if rounds_path.exists():
        rounds_path.unlink()
    per_client: list[dict[str, Any]] = []
    start = time.perf_counter()
    for state in clients:
        vec = init_vec.clone()
        for r in range(1, rounds + 1):
            update = trainer.train(
                state, vec, r, round_lr(cfg["client"], r, rounds), seed=seed + 100_003 * r + 1_009 * state.client_id
            )
            vec = vec + update.delta
        trainer.model.load_state_dict(layout.unflatten(vec))
        val = evaluator.evaluate(val_ids)
        test = evaluator.evaluate(ctx.test_ids)
        per_client.append({"client_id": state.client_id, "num_flows": int(state.flow_ids.shape[0]), "val": val, "test": test})
        append_jsonl(rounds_path, {"client_id": state.client_id, "val": compact_metrics(val), "test": compact_metrics(test)})
        print(f"[local_only] client {state.client_id} test_macro_f1={test.get('macro_f1')}", flush=True)

    def stats(split: str, clean: bool = False) -> dict[str, float | None]:
        values = []
        for row in per_client:
            m = row[split].get("clean") if clean else row[split]
            if m and m.get("macro_f1") is not None:
                values.append(float(m["macro_f1"]))
        if not values:
            return {"mean": None, "std": None, "min": None, "max": None}
        arr = np.asarray(values)
        return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}

    return {
        "experiment": cfg["experiment"].get("name"),
        "labels": cfg["experiment"].get("labels") or {},
        "mode": mode,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "num_clients": len(clients),
        "client_sizes": partition.sizes(),
        "malicious_clients": [],
        "attack": "none",
        "aggregator": None,
        "screen": None,
        "num_params": layout.numel,
        "rounds": rounds,
        "per_client": per_client,
        "aggregate": {
            "val_macro_f1": stats("val"),
            "val_macro_f1_clean": stats("val", clean=True),
            "test_macro_f1": stats("test"),
            "test_macro_f1_clean": stats("test", clean=True),
        },
        "cost": {"wall_seconds": time.perf_counter() - start},
        "config": cfg,
    }


def run_experiment(
    cfg: dict[str, Any], *, out_dir: str | Path | None = None, context: ExperimentContext | None = None
) -> dict[str, Any]:
    mode = str(cfg["federated"].get("mode", "fl")).lower()
    if mode not in VALID_MODES:
        raise ValueError(f"unknown federated.mode {mode!r}; expected one of {VALID_MODES}")
    monitor = str(cfg["eval"].get("monitor", "val_macro_f1"))
    if monitor not in VALID_MONITORS:
        raise ValueError(f"unknown eval.monitor {monitor!r}; expected one of {VALID_MONITORS}")
    if context is not None and context.data_key != data_key(cfg):
        raise ValueError("experiment context was loaded for a different data section")
    if mode == "centralized":
        cfg = apply_overrides(
            cfg,
            {
                "client.optimizer_state": "persist",
                "attack.name": "none",
                "attack.malicious_ratio": 0.0,
                "defense.clip": {"mode": "none"},
                "defense.screen": {"name": "none", "params": {}},
                "defense.aggregator": {"name": "fedavg", "params": {}},
                "defense.relation_aware": False,
                "federated.participation": 1.0,
            },
        )
    elif mode == "local_only":
        cfg = apply_overrides(cfg, {"client.optimizer_state": "persist"})
    seed = int(cfg["federated"].get("seed", 42))
    set_seed(seed)
    _set_tew_state(False)
    device = resolve_device(str(cfg.get("device", "auto")))
    out = Path(out_dir or cfg["output"]["dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.resolved.yaml").write_text(yaml.safe_dump(to_jsonable(cfg), sort_keys=False), encoding="utf-8")
    ctx = context if context is not None else load_context(cfg)

    partition = make_partition(ctx.train_ids, ctx.labels, cfg["partition"])
    if mode == "centralized":
        # Same (possibly subsampled) training flows as the FL clients, one silo.
        partition = ClientPartition([partition.all_flow_ids()], "centralized", None, seed)
    write_json(
        out / "partition.json",
        {
            **partition.to_json(),
            "sizes": partition.sizes(),
            "class_histogram": partition.class_histogram(ctx.labels, ctx.num_classes),
        },
    )
    flow_stats = federated_flow_stats(ctx.backend, partition.client_flow_ids)
    clients = build_clients(cfg, ctx, partition, flow_stats, build_base_eacs(cfg, ctx))
    torch.manual_seed(seed)
    model = build_model(cfg, ctx.backend, ctx.num_classes).to(device)
    layout = ParamLayout.from_model(model)
    init_vec = layout.flatten(model.state_dict())
    trainer = LocalTrainer(
        model,
        layout,
        ctx.backend.edge_types,
        device,
        cfg["client"],
        use_semantic_edge_weights=bool(cfg["data"].get("use_semantic_edge_weights", True)),
        gcl_map=build_gcl_map(cfg, ctx.backend),
    )
    evaluator = Evaluator(cfg, ctx, model, flow_stats, device)
    val_ids = stratified_subset(ctx.val_ids, ctx.labels, cfg["eval"].get("max_val_flows"), seed)
    runner = _run_local_only if mode == "local_only" else _run_federated
    summary = runner(
        cfg=cfg,
        ctx=ctx,
        out=out,
        clients=clients,
        trainer=trainer,
        evaluator=evaluator,
        layout=layout,
        init_vec=init_vec,
        val_ids=val_ids,
        flow_stats=flow_stats,
        partition=partition,
        device=device,
        mode=mode,
    )
    write_json(out / "fl_summary.json", summary)
    return to_jsonable(summary)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PY -m pytest tests/federated/test_simulation.py -q`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/federated/conftest.py src/graphslm_ids/offline/federated/simulation.py tests/federated/test_simulation.py
git commit -m "feat(fl): add FL experiment driver with fl, centralized and local_only modes"
```

---

### Task 13: Command-line entry point (`cli.py`)

**Files:**
- Create: `src/graphslm_ids/offline/federated/cli.py`
- Test: `tests/federated/test_cli.py`

- [ ] **Step 1: Write the failing tests**

```python
import json

import pytest
import yaml

from graphslm_ids.offline.federated.cli import main


def test_cli_runs_one_experiment(smoke_cfg, tmp_path, capsys):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(smoke_cfg("cli")), encoding="utf-8")
    code = main(
        ["--config", str(path), "--rounds", "1", "--set", "attack.name=sign_flip", "--set", "attack.malicious_ratio=0.34"]
    )
    assert code == 0
    assert '"attack": "sign_flip"' in capsys.readouterr().out
    summary = json.loads((tmp_path / "cli" / "fl_summary.json").read_text(encoding="utf-8"))
    assert summary["rounds"] == 1


def test_cli_rejects_malformed_set(smoke_cfg, tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(smoke_cfg("bad")), encoding="utf-8")
    with pytest.raises(ValueError):
        main(["--config", str(path), "--set", "attack.name"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PY -m pytest tests/federated/test_cli.py -q`
Expected: FAIL — `ModuleNotFoundError: ... federated.cli`

- [ ] **Step 3: Write the implementation**

```python
"""Run one FL-HGT poisoning experiment.

    python -m graphslm_ids.offline.federated.cli --config configs/fl/fl_hgt_screen.yaml \
        --device cuda --set attack.name=alie --set attack.malicious_ratio=0.2
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from graphslm_ids.offline.federated.config import apply_overrides, load_fl_config, parse_set_args
from graphslm_ids.offline.federated.records import to_jsonable
from graphslm_ids.offline.federated.simulation import VALID_MODES, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one FL-HGT poisoning experiment.")
    parser.add_argument("--config", required=True, help="FL YAML config (configs/fl/*.yaml)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="dotted override, repeatable")
    parser.add_argument("--device", help="cpu | cuda | auto")
    parser.add_argument("--output-dir")
    parser.add_argument("--mode", choices=list(VALID_MODES))
    parser.add_argument("--rounds", type=int)
    return parser


def headline(summary: dict[str, Any]) -> dict[str, Any]:
    final = (summary.get("final") or {}).get("test") or {}
    return {
        "experiment": summary.get("experiment"),
        "mode": summary.get("mode"),
        "attack": summary.get("attack"),
        "aggregator": summary.get("aggregator"),
        "screen": summary.get("screen"),
        "test_macro_f1": final.get("macro_f1"),
        "test_macro_f1_clean": (final.get("clean") or {}).get("macro_f1"),
        "detection": summary.get("detection"),
        "aggregate": summary.get("aggregate"),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = parse_set_args(args.set)
    if args.device:
        overrides["device"] = args.device
    if args.output_dir:
        overrides["output.dir"] = args.output_dir
    if args.mode:
        overrides["federated.mode"] = args.mode
    if args.rounds:
        overrides["federated.rounds"] = args.rounds
    cfg = apply_overrides(load_fl_config(args.config), overrides)
    summary = run_experiment(cfg)
    print(json.dumps(to_jsonable(headline(summary)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PY -m pytest tests/federated/test_cli.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/graphslm_ids/offline/federated/cli.py tests/federated/test_cli.py
git commit -m "feat(fl): add single-experiment CLI"
```

---

### Task 14: Experiment matrix (`matrix.py` + scripts)

**Files:**
- Create: `src/graphslm_ids/offline/federated/matrix.py`
- Create: `scripts/fl/run_matrix.py`
- Create: `scripts/fl/summarize_matrix.py`
- Test: `tests/federated/test_matrix.py`

- [ ] **Step 1: Write the failing tests**

```python
import importlib.util
from pathlib import Path

import yaml

from graphslm_ids.offline.federated.matrix import (
    expand_grid,
    pivot_markdown,
    run_grid,
    summarize,
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


def test_run_grid_runs_resumes_and_summarizes(smoke_cfg, tmp_path):
    grid_path = _grid(smoke_cfg, tmp_path)
    assert [r["status"] for r in run_grid(grid_path, dry_run=True)] == ["planned", "planned"]
    assert [r["status"] for r in run_grid(grid_path)] == ["done", "done"]
    assert [r["status"] for r in run_grid(grid_path)] == ["skipped", "skipped"]
    assert not (tmp_path / "matrix" / "none__m0__fedavg" / "global_final.pt").exists()
    rows = {r["name"]: r for r in summarize(tmp_path / "matrix")}
    assert rows["none__m0__fedavg"]["impact_final_test_macro_f1"] == 0.0
    assert rows["sign_flip__m0.34__fedavg"]["impact_final_test_macro_f1"] is not None
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
    assert (tmp_path / "matrix" / "matrix_results.md").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PY -m pytest tests/federated/test_matrix.py -q`
Expected: FAIL — `ModuleNotFoundError: ... federated.matrix`

- [ ] **Step 3: Write the implementation**

`src/graphslm_ids/offline/federated/matrix.py`:

```python
"""Experiment grid: expand axes, run cells sequentially (resumable), summarize.

Grid YAML: ``base_config``, ``output_root``, ``name_template``, ``overrides`` (applied
to every cell) and ``axes``: {axis: {label: {dotted.key: value}}}. A cell whose
``attack.name`` is ``none`` gets ``attack.malicious_ratio = 0`` and is deduplicated.
"""

from __future__ import annotations

import csv
import fnmatch
import itertools
import json
import traceback
from pathlib import Path
from typing import Any

import yaml

from graphslm_ids.offline.federated.config import apply_overrides, load_fl_config
from graphslm_ids.offline.federated.simulation import data_key, load_context, run_experiment

IMPACT_METRICS = ("final_test_macro_f1", "final_test_macro_f1_clean")


def expand_grid(grid: dict[str, Any]) -> list[dict[str, Any]]:
    axes = grid.get("axes") or {}
    names = list(axes)
    template = grid.get("name_template") or "__".join("{" + n + "}" for n in names)
    cells: list[dict[str, Any]] = []
    seen: set[str] = set()
    for combo in itertools.product(*[list((axes[n] or {}).items()) for n in names]):
        labels = {n: str(label) for n, (label, _ov) in zip(names, combo)}
        overrides: dict[str, Any] = {}
        for _label, ov in combo:
            overrides.update(ov or {})
        if str(overrides.get("attack.name", "")).lower() == "none":
            overrides["attack.malicious_ratio"] = 0.0
            if "malicious_ratio" in labels:
                labels["malicious_ratio"] = "0"
        name = template.format(**labels)
        if name in seen:
            continue
        seen.add(name)
        cells.append({"name": name, "labels": labels, "overrides": overrides})
    # Group cells that share a partition so client graphs are built once per group.
    cells.sort(key=lambda c: (c["labels"].get("partition", ""), c["labels"].get("seed", "")))
    return cells


def run_grid(
    grid_path: str | Path,
    *,
    device: str | None = None,
    dry_run: bool = False,
    only: str | None = None,
    fail_fast: bool = False,
) -> list[dict[str, str]]:
    grid = yaml.safe_load(Path(grid_path).read_text(encoding="utf-8")) or {}
    base = load_fl_config(grid["base_config"])
    root = Path(grid.get("output_root", "outputs/fl/matrix"))
    common = dict(grid.get("overrides") or {})
    if device:
        common["device"] = device
    cells = expand_grid(grid)
    if only:
        cells = [c for c in cells if fnmatch.fnmatch(c["name"], only)]
    results: list[dict[str, str]] = []
    ctx = None
    for i, cell in enumerate(cells, start=1):
        out = root / cell["name"]
        if (out / "fl_summary.json").exists():
            results.append({"name": cell["name"], "status": "skipped"})
            continue
        if dry_run:
            print(f"[matrix] {i}/{len(cells)} {cell['name']}")
            results.append({"name": cell["name"], "status": "planned"})
            continue
        cfg = apply_overrides(
            base,
            {
                **common,
                **cell["overrides"],
                "output.dir": str(out),
                "experiment.name": cell["name"],
                "experiment.labels": cell["labels"],
            },
        )
        print(f"[matrix] {i}/{len(cells)} {cell['name']}", flush=True)
        try:
            if ctx is None or ctx.data_key != data_key(cfg):
                ctx = load_context(cfg)
            run_experiment(cfg, out_dir=out, context=ctx)
            results.append({"name": cell["name"], "status": "done"})
        except Exception:
            out.mkdir(parents=True, exist_ok=True)
            (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            results.append({"name": cell["name"], "status": "failed"})
            if fail_fast:
                raise
    return results


def _get(d: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(key)
    return d


def summary_row(summary: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": summary.get("experiment"),
        "mode": summary.get("mode"),
        "attack": summary.get("attack"),
        "aggregator": summary.get("aggregator"),
        "screen": summary.get("screen"),
    }
    row.update({f"label_{k}": v for k, v in (summary.get("labels") or {}).items()})
    if summary.get("mode") == "local_only":
        row["final_test_macro_f1"] = _get(summary, "aggregate", "test_macro_f1", "mean")
        row["final_test_macro_f1_clean"] = _get(summary, "aggregate", "test_macro_f1_clean", "mean")
        return row
    row.update(
        {
            "final_test_macro_f1": _get(summary, "final", "test", "macro_f1"),
            "final_test_macro_f1_clean": _get(summary, "final", "test", "clean", "macro_f1"),
            "best_test_macro_f1": _get(summary, "best", "test", "macro_f1"),
            "best_test_macro_f1_clean": _get(summary, "best", "test", "clean", "macro_f1"),
            "final_test_benign_far": _get(summary, "final", "test", "benign_false_alarm_rate"),
            "final_test_attack_detection": _get(summary, "final", "test", "attack_detection_rate"),
            "final_test_web_recall": _get(summary, "final", "test", "family_recall", "web"),
            "detect_tpr": _get(summary, "detection", "tpr"),
            "benign_false_rejection": _get(summary, "detection", "fpr"),
            "malicious_acceptance": _get(summary, "detection", "malicious_acceptance_rate"),
            "convergence_round": summary.get("convergence_round"),
            "mean_round_seconds": _get(summary, "cost", "mean_round_seconds"),
            "peak_vram_gb": _get(summary, "cost", "peak_vram_gb"),
            "peak_rss_gb": _get(summary, "cost", "peak_rss_gb"),
        }
    )
    return row


def add_attack_impact(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """impact = metric(no attack, same partition/defense/seed) - metric(this cell)."""

    def key(r: dict[str, Any]) -> tuple:
        return (r.get("label_partition"), r.get("label_defense"), r.get("label_seed"))

    reference = {key(r): r for r in rows if r.get("attack") == "none" and r.get("mode") == "fl"}
    for r in rows:
        base = reference.get(key(r))
        for metric in IMPACT_METRICS:
            ok = base is not None and base.get(metric) is not None and r.get(metric) is not None
            r[f"impact_{metric}"] = (base[metric] - r[metric]) if ok else None
    return rows


def summarize(root: str | Path) -> list[dict[str, Any]]:
    rows = [
        summary_row(json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(Path(root).rglob("fl_summary.json"))
    ]
    return add_attack_impact(rows)


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pivot_markdown(
    rows: list[dict[str, Any]],
    value: str = "final_test_macro_f1",
    row_keys: tuple[str, ...] = ("label_attack", "label_malicious_ratio"),
    col_key: str = "label_defense",
    group_key: str = "label_partition",
) -> str:
    lines: list[str] = []
    for group in sorted({str(r.get(group_key)) for r in rows}):
        sub = [r for r in rows if str(r.get(group_key)) == group]
        cols = sorted({str(r.get(col_key)) for r in sub})
        keys = sorted({tuple(str(r.get(k)) for k in row_keys) for r in sub})
        cell = {(tuple(str(r.get(k)) for k in row_keys), str(r.get(col_key))): r.get(value) for r in sub}
        lines.append(f"### {group_key.removeprefix('label_')} = {group} — {value}")
        lines.append("")
        lines.append("| " + " | ".join([*(k.removeprefix("label_") for k in row_keys), *cols]) + " |")
        lines.append("|" + "---|" * (len(row_keys) + len(cols)))
        for rk in keys:
            vals = [cell.get((rk, c)) for c in cols]
            lines.append("| " + " | ".join([*rk, *("" if v is None else f"{v:.4f}" for v in vals)]) + " |")
        lines.append("")
    return "\n".join(lines)
```

`scripts/fl/run_matrix.py`:

```python
"""Run an FL experiment grid (resumable: finished cells are skipped).

    python scripts/fl/run_matrix.py --grid configs/fl/matrix_screen.yaml --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from graphslm_ids.offline.federated.matrix import run_grid  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an FL experiment grid.")
    parser.add_argument("--grid", required=True)
    parser.add_argument("--device")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", help="fnmatch pattern on cell names")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)
    results = run_grid(args.grid, device=args.device, dry_run=args.dry_run, only=args.only, fail_fast=args.fail_fast)
    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"[matrix] {len(results)} cells: {counts}")
    return 1 if counts.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

`scripts/fl/summarize_matrix.py`:

```python
"""Summarize an FL matrix run into CSV + markdown pivot tables.

    python scripts/fl/summarize_matrix.py --root outputs/fl/matrix_screen
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from graphslm_ids.offline.federated.matrix import pivot_markdown, summarize, write_csv  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize an FL experiment grid.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--out-csv")
    parser.add_argument("--out-md")
    parser.add_argument("--value", action="append", help="metric column to pivot (repeatable)")
    args = parser.parse_args(argv)
    root = Path(args.root)
    rows = summarize(root)
    write_csv(rows, Path(args.out_csv or root / "matrix_results.csv"))
    values = args.value or ["final_test_macro_f1", "final_test_macro_f1_clean", "benign_false_rejection", "detect_tpr"]
    md = "\n".join(pivot_markdown(rows, value=v) for v in values)
    Path(args.out_md or root / "matrix_results.md").write_text(md, encoding="utf-8")
    print(f"[summarize] {len(rows)} runs -> {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PY -m pytest tests/federated/test_matrix.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/graphslm_ids/offline/federated/matrix.py scripts/fl/run_matrix.py scripts/fl/summarize_matrix.py tests/federated/test_matrix.py
git commit -m "feat(fl): add resumable experiment matrix runner and summarizer"
```

---

### Task 15: Experiment configs (`configs/fl/`)

**Files:**
- Create: `configs/fl/fl_hgt_screen.yaml`, `configs/fl/fl_hgt_full.yaml`
- Create: `configs/fl/matrix_screen.yaml`, `configs/fl/matrix_full.yaml`, `configs/fl/matrix_baselines.yaml`
- Test: `tests/federated/test_configs.py`

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

import pytest
import yaml

from graphslm_ids.offline.federated.aggregators import build_aggregator
from graphslm_ids.offline.federated.attacks import build_attack
from graphslm_ids.offline.federated.config import apply_overrides, load_fl_config
from graphslm_ids.offline.federated.matrix import expand_grid
from graphslm_ids.offline.federated.screening import build_screener
from graphslm_ids.offline.federated.simulation import VALID_MODES

ROOT = Path(__file__).resolve().parents[2]
CFG_DIR = ROOT / "configs" / "fl"
GRID_SIZES = {"matrix_screen.yaml": 576, "matrix_full.yaml": 48, "matrix_baselines.yaml": 9}


def test_tier_configs_load():
    for name in ("fl_hgt_screen.yaml", "fl_hgt_full.yaml"):
        cfg = load_fl_config(CFG_DIR / name)
        assert cfg["data"]["artifact_version"] == "v3"
        assert cfg["federated"]["mode"] in VALID_MODES
        assert cfg["client"]["eacs"]["enabled"] and cfg["client"]["gcl"]["enabled"]


@pytest.mark.parametrize("grid_name", sorted(GRID_SIZES))
def test_every_matrix_cell_builds(grid_name):
    grid = yaml.safe_load((CFG_DIR / grid_name).read_text(encoding="utf-8"))
    base = load_fl_config(ROOT / grid["base_config"])
    cells = expand_grid(grid)
    assert len(cells) == GRID_SIZES[grid_name]
    for cell in cells:
        cfg = apply_overrides(base, {**(grid.get("overrides") or {}), **cell["overrides"]})
        assert cfg["federated"]["mode"] in VALID_MODES
        build_attack(cfg["attack"])
        build_aggregator(cfg["defense"]["aggregator"], assumed_malicious_ratio=cfg["defense"]["assumed_malicious_ratio"])
        build_screener(cfg["defense"]["screen"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PY -m pytest tests/federated/test_configs.py -q`
Expected: FAIL — `FileNotFoundError` for `configs/fl/fl_hgt_screen.yaml`

- [ ] **Step 3: Write the configs**

`configs/fl/fl_hgt_screen.yaml`:

```yaml
# FL-HGT poisoning testbed — SCREEN tier (cheap enough for the full matrix).
# Design: docs/superpowers/specs/2026-10-02-fl-hgt-poisoning-testbed-design.md
# Smaller HGT (64/2/4), 2-hop sampling with fanout 5-10 per relation (proposal),
# 10 Dirichlet(0.5) clients capped at 4000 stratified flows each.
experiment:
  name: fl_screen

data:
  artifact_version: v3
  graph_npz: outputs/v3_ob/graph.npz
  graph_meta_json: outputs/v3_ob/graph.meta.json
  splits_json: outputs/v3_ob/splits.json
  split_protocol: random
  packet_dtype: preserve
  add_reverse_edges: true
  use_semantic_edge_weights: true
  standardize_flow_features: true
  clean_eval_labels: outputs/v3_ob/clean_eval_labels.npy

model:
  hidden_dim: 64
  num_layers: 2
  num_heads: 4
  dropout: 0.2
  ffn_multiplier: 2

sampler:
  hops: 2
  fanouts:
    flow__contains__packet: 10
    packet__next_packet__packet: 5
    packet__evidence_injection__technique: 5
    packet__evidence_command_exec__technique: 5
    packet__evidence_file_upload__technique: 5
    packet__evidence_recon__technique: 5
    packet__evidence_c2_beacon__technique: 5
    flow__from_host__host: 1
    flow__to_host__host: 1
    flow__burst_neighbor__flow: 5
    flow__matches_technique__technique: 5
    technique__has_subtechnique__technique: 5
    technique__belongs_to_tactic__tactic: 5
  reverse_fanouts:
    rev_contains: 5
    rev_next_packet: 5
    rev_evidence_injection: 5
    rev_evidence_command_exec: 5
    rev_evidence_file_upload: 5
    rev_evidence_recon: 5
    rev_evidence_c2_beacon: 5
    rev_from_host: 5
    rev_to_host: 5
    rev_burst_neighbor: 5
    rev_matches_technique: 5
    rev_has_subtechnique: 5
    rev_belongs_to_tactic: 5
  always_include_all_techniques: true
  always_include_all_tactics: true

partition:
  scheme: dirichlet
  num_clients: 10
  alpha: 0.5
  min_flows_per_client: 100
  max_flows_per_client: 4000
  seed: 42

client:
  local_epochs: 1
  max_local_steps: null
  batch_seed_flows: 256
  lr: 1.0e-3
  lr_schedule: cosine_rounds
  lr_min: 1.0e-5
  weight_decay: 5.0e-5
  adamw_eps: 1.0e-6
  adamw_betas: [0.9, 0.95]
  grad_clip_norm: 1.0
  optimizer_state: reset
  loss_type: focal
  focal_gamma: 1.5
  label_smoothing: 0.05
  class_weight: balanced
  class_weight_method: cb
  cb_beta: 0.985
  class_weight_cap: 4.0
  stratified_batch_sampling: true
  drop_edge_prob: 0.1
  num_workers: 0
  pin_memory: false
  gcl:
    enabled: true
    weight: 0.3
    temperature: 0.1
    n_negatives: 16
    class_technique_map_csv: data/mitre/class_technique_map.csv
  eacs:
    enabled: true
    anchor_mask_npy: outputs/v3_ob/eacs_anchor_mask.npy
    warmup_rounds: 5
    ema_decay: 0.9
    lambda_disambig: 0.7
    suspect_classes: [CommandInjection, XSS, SqlInjection, Uploading_Attack]
    mitre_dir: data/mitre

federated:
  mode: fl
  rounds: 30
  participation: 1.0
  server_lr: 1.0
  seed: 42
  cache_clients: true

attack:
  name: none
  malicious_ratio: 0.0
  knowledge: omniscient
  start_round: 1
  params: {}

defense:
  clip: {mode: none, factor: 1.0, value: null}
  screen: {name: none, params: {}}
  aggregator: {name: fedavg, params: {}}
  assumed_malicious_ratio: 0.2
  relation_aware: false
  weighting: samples

eval:
  every: 1
  max_val_flows: 5000
  monitor: val_macro_f1_clean
  batch_seed_flows: 512

output:
  dir: outputs/fl/screen_single
  save_checkpoints: true

device: cuda
```

`configs/fl/fl_hgt_full.yaml`:

```yaml
# FL-HGT poisoning testbed — FULL tier (headline cells; model + sampler as EACS v2,
# minus HGAA/TEW). num_workers > 0 assumes Linux fork on the L40S server.
experiment:
  name: fl_full

data:
  artifact_version: v3
  graph_npz: outputs/v3_ob/graph.npz
  graph_meta_json: outputs/v3_ob/graph.meta.json
  splits_json: outputs/v3_ob/splits.json
  split_protocol: random
  packet_dtype: preserve
  add_reverse_edges: true
  use_semantic_edge_weights: true
  standardize_flow_features: true
  clean_eval_labels: outputs/v3_ob/clean_eval_labels.npy

model:
  hidden_dim: 128
  num_layers: 4
  num_heads: 8
  dropout: 0.2
  ffn_multiplier: 2

sampler:
  hops: 4
  fanouts:
    flow__contains__packet: 20
    packet__next_packet__packet: 4
    packet__evidence_injection__technique: 4
    packet__evidence_command_exec__technique: 4
    packet__evidence_file_upload__technique: 3
    packet__evidence_recon__technique: 3
    packet__evidence_c2_beacon__technique: 3
    flow__from_host__host: 1
    flow__to_host__host: 1
    flow__burst_neighbor__flow: 6
    flow__matches_technique__technique: 3
    technique__has_subtechnique__technique: 2
    technique__belongs_to_tactic__tactic: 1
  reverse_fanouts:
    rev_contains: 3
    rev_next_packet: 2
    rev_evidence_injection: 2
    rev_evidence_command_exec: 2
    rev_evidence_file_upload: 2
    rev_evidence_recon: 2
    rev_evidence_c2_beacon: 2
    rev_from_host: 3
    rev_to_host: 3
    rev_burst_neighbor: 3
    rev_matches_technique: 2
    rev_has_subtechnique: 2
    rev_belongs_to_tactic: 1
  always_include_all_techniques: true
  always_include_all_tactics: true

partition:
  scheme: dirichlet
  num_clients: 10
  alpha: 0.5
  min_flows_per_client: 100
  max_flows_per_client: null
  seed: 42

client:
  local_epochs: 1
  max_local_steps: null
  batch_seed_flows: 256
  lr: 1.0e-3
  lr_schedule: cosine_rounds
  lr_min: 1.0e-5
  weight_decay: 5.0e-5
  adamw_eps: 1.0e-6
  adamw_betas: [0.9, 0.95]
  grad_clip_norm: 1.0
  optimizer_state: reset
  loss_type: focal
  focal_gamma: 1.5
  label_smoothing: 0.05
  class_weight: balanced
  class_weight_method: cb
  cb_beta: 0.985
  class_weight_cap: 4.0
  stratified_batch_sampling: true
  drop_edge_prob: 0.1
  num_workers: 4
  pin_memory: true
  gcl:
    enabled: true
    weight: 0.3
    temperature: 0.1
    n_negatives: 16
    class_technique_map_csv: data/mitre/class_technique_map.csv
  eacs:
    enabled: true
    anchor_mask_npy: outputs/v3_ob/eacs_anchor_mask.npy
    warmup_rounds: 5
    ema_decay: 0.9
    lambda_disambig: 0.7
    suspect_classes: [CommandInjection, XSS, SqlInjection, Uploading_Attack]
    mitre_dir: data/mitre

federated:
  mode: fl
  rounds: 50
  participation: 1.0
  server_lr: 1.0
  seed: 42
  cache_clients: true

attack:
  name: none
  malicious_ratio: 0.0
  knowledge: omniscient
  start_round: 1
  params: {}

defense:
  clip: {mode: none, factor: 1.0, value: null}
  screen: {name: none, params: {}}
  aggregator: {name: fedavg, params: {}}
  assumed_malicious_ratio: 0.2
  relation_aware: false
  weighting: samples

eval:
  every: 2
  max_val_flows: null
  monitor: val_macro_f1_clean
  batch_seed_flows: 512

output:
  dir: outputs/fl/full_single
  save_checkpoints: true

device: cuda
```

`configs/fl/matrix_screen.yaml` (3 partitions x (1 + 5 attacks x 3 ratios) x 12 defenses = 576 cells):

```yaml
base_config: configs/fl/fl_hgt_screen.yaml
output_root: outputs/fl/matrix_screen
name_template: "{partition}__{attack}__m{malicious_ratio}__{defense}__s{seed}"
overrides:
  output.save_checkpoints: false
axes:
  partition:
    iid: {partition.scheme: iid}
    dir0.5: {partition.scheme: dirichlet, partition.alpha: 0.5}
    dir0.1: {partition.scheme: dirichlet, partition.alpha: 0.1}
  attack:
    none: {attack.name: none}
    sign_flip: {attack.name: sign_flip}
    alie: {attack.name: alie}
    min_max: {attack.name: min_max}
    min_sum: {attack.name: min_sum}
    scale_sign: {attack.name: scale_sign}
  malicious_ratio:
    "0.1": {attack.malicious_ratio: 0.1}
    "0.2": {attack.malicious_ratio: 0.2}
    "0.3": {attack.malicious_ratio: 0.3}
  defense:
    fedavg: {defense.aggregator: {name: fedavg}}
    median: {defense.aggregator: {name: median}}
    trimmed_mean: {defense.aggregator: {name: trimmed_mean}}
    multi_krum: {defense.aggregator: {name: multi_krum}}
    signguard: {defense.aggregator: {name: signguard}}
    msguard: {defense.aggregator: {name: msguard}}
    dnc: {defense.aggregator: {name: dnc}}
    clip_median: {defense.clip: {mode: median, factor: 1.0}, defense.aggregator: {name: median}}
    temporal_audit: {defense.screen: {name: temporal_audit, params: {}}}
    multi_signal: {defense.screen: {name: multi_signal, params: {}}}
    bray_curtis: {defense.screen: {name: bray_curtis, params: {variant: abs}}}
    bray_curtis_signed: {defense.screen: {name: bray_curtis, params: {variant: signed}}}
  seed:
    "42": {federated.seed: 42, partition.seed: 42}
```

`configs/fl/matrix_full.yaml` (1 partition x (1 + 5 attacks) x 8 defenses = 48 cells):

```yaml
base_config: configs/fl/fl_hgt_full.yaml
output_root: outputs/fl/matrix_full
name_template: "{partition}__{attack}__m{malicious_ratio}__{defense}__s{seed}"
overrides:
  output.save_checkpoints: true
axes:
  partition:
    dir0.5: {partition.scheme: dirichlet, partition.alpha: 0.5}
  attack:
    none: {attack.name: none}
    sign_flip: {attack.name: sign_flip}
    alie: {attack.name: alie}
    min_max: {attack.name: min_max}
    min_sum: {attack.name: min_sum}
    scale_sign: {attack.name: scale_sign}
  malicious_ratio:
    "0.2": {attack.malicious_ratio: 0.2}
  defense:
    fedavg: {defense.aggregator: {name: fedavg}}
    trimmed_mean: {defense.aggregator: {name: trimmed_mean}}
    multi_krum: {defense.aggregator: {name: multi_krum}}
    signguard: {defense.aggregator: {name: signguard}}
    msguard: {defense.aggregator: {name: msguard}}
    temporal_audit: {defense.screen: {name: temporal_audit, params: {}}}
    multi_signal: {defense.screen: {name: multi_signal, params: {}}}
    bray_curtis_signed: {defense.screen: {name: bray_curtis, params: {variant: signed}}}
  seed:
    "42": {federated.seed: 42, partition.seed: 42}
```

`configs/fl/matrix_baselines.yaml` (centralized / local-only / FedAvg x 3 partitions = 9 cells):

```yaml
base_config: configs/fl/fl_hgt_screen.yaml
output_root: outputs/fl/matrix_baselines
name_template: "{mode}__{partition}"
overrides:
  attack.name: none
  output.save_checkpoints: true
axes:
  mode:
    centralized: {federated.mode: centralized}
    local_only: {federated.mode: local_only}
    fedavg: {federated.mode: fl}
  partition:
    iid: {partition.scheme: iid}
    dir0.5: {partition.scheme: dirichlet, partition.alpha: 0.5}
    dir0.1: {partition.scheme: dirichlet, partition.alpha: 0.1}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PY -m pytest tests/federated/test_configs.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add configs/fl tests/federated/test_configs.py
git commit -m "feat(fl): add screen/full tier configs and experiment grids"
```

---

### Task 16: Documentation and full verification

**Files:**
- Modify: `CLAUDE.md` (append an FL section after "Tiered feature store + GPU sampling")
- Modify: `docs/superpowers/specs/2026-10-02-fl-hgt-poisoning-testbed-design.md` (only if implementation deviated)

- [ ] **Step 1: Append the FL section to `CLAUDE.md`**

Insert this block immediately before the line `## Diagnosis history (v1, May 2026)`:

````markdown
## Federated-learning poisoning testbed (FL-HGT)

**Design:** `docs/superpowers/specs/2026-10-02-fl-hgt-poisoning-testbed-design.md`
**Plan:** `docs/superpowers/plans/2026-10-02-fl-hgt-poisoning-testbed.md`

Package `src/graphslm_ids/offline/federated/` simulates K clients on one v3 artifact:
client-local graph views (edges kept only when every flow/packet endpoint belongs to
the client; host features recomputed locally), lean local HGT training (focal + CB from
local labels, GCL, EACS), deltas only to the server, five model-poisoning attacks
(sign-flip, ALIE, Min-Max, Min-Sum, ScaleSign), robust aggregators (FedAvg, Median,
Trimmed Mean, Multi-Krum, SignGuard, MSGuard, DnC, clipping, relation-aware) and three
delta-only screeners (TemporalAudit, MultiSignal, Bray–Curtis). Modes: `fl`,
`centralized` (same flows, one silo), `local_only`. It does not modify
`train_hgt_flow_classifier.py`.

```bat
:: One experiment (screen tier)
D:\v\nt114\Scripts\python.exe -m graphslm_ids.offline.federated.cli ^
  --config configs/fl/fl_hgt_screen.yaml --device cuda ^
  --set attack.name=alie --set attack.malicious_ratio=0.2 ^
  --set defense.screen.name=temporal_audit

:: Baselines (centralized / local-only / FedAvg x IID, Dir(0.5), Dir(0.1))
D:\v\nt114\Scripts\python.exe scripts/fl/run_matrix.py --grid configs/fl/matrix_baselines.yaml --device cuda

:: Full screen matrix (576 cells, resumable) and headline matrix (48 cells)
D:\v\nt114\Scripts\python.exe scripts/fl/run_matrix.py --grid configs/fl/matrix_screen.yaml --device cuda
D:\v\nt114\Scripts\python.exe scripts/fl/run_matrix.py --grid configs/fl/matrix_full.yaml --device cuda

:: Tables (CSV + markdown pivots, attack impact vs the no-attack cell)
D:\v\nt114\Scripts\python.exe scripts/fl/summarize_matrix.py --root outputs/fl/matrix_screen

:: FL unit + smoke tests (CPU)
D:\v\nt114\Scripts\python.exe -m pytest tests/federated -q
```

Per run: `outputs/fl/<run>/{config.resolved.yaml, partition.json, rounds.jsonl,
fl_summary.json, global_final.pt, global_best.pt}`; checkpoints load in
`HGTRuntime`. Report the **final-round** model under poisoning (best-val uses the
experiment-only evaluator). TemporalAudit and MSGuard are reconstructions where the
papers omit details; BrayCurtis `abs` is blind to pure sign flips (use `signed` as the
ablation).
````

- [ ] **Step 2: Run the whole FL suite**

Run: `PY -m pytest tests/federated -q`
Expected: `97 passed`

- [ ] **Step 3: Run the full repository suite**

Run: `PY -m pytest tests/ -q`
Expected: `437 passed, 3 skipped` (baseline before this branch: 340 passed, 3 skipped)

- [ ] **Step 4: Smoke the CLI end to end on the tiny fixture outside pytest**

```bash
PY - <<'EOF'
import sys, tempfile, pathlib
sys.path.insert(0, "src"); sys.path.insert(0, "tests")
from federated.conftest import write_tiny_v3
from graphslm_ids.offline.federated.config import apply_overrides, load_fl_config
from graphslm_ids.offline.federated.simulation import run_experiment
root = pathlib.Path(tempfile.mkdtemp())
p = write_tiny_v3(root / "tiny")
cfg = apply_overrides(load_fl_config("configs/fl/fl_hgt_screen.yaml"), {
    "data.graph_npz": str(p["graph_npz"]), "data.graph_meta_json": str(p["graph_meta_json"]),
    "data.splits_json": str(p["splits_json"]), "data.clean_eval_labels": str(p["clean_eval_labels"]),
    "client.eacs.anchor_mask_npy": str(p["anchor_mask_npy"]), "client.eacs.suspect_classes": ["XSS"],
    "client.gcl.class_technique_map_csv": str(p["class_technique_map_csv"]),
    "partition.scheme": "iid", "partition.num_clients": 3, "partition.max_flows_per_client": None,
    "federated.rounds": 2, "attack.name": "scale_sign", "attack.malicious_ratio": 0.34,
    "defense.aggregator": {"name": "signguard"}, "eval.max_val_flows": None,
    "output.dir": str(root / "run"), "device": "cpu"})
s = run_experiment(cfg)
print("OK", s["attack"], s["aggregator"], s["final"]["test"]["macro_f1"])
EOF
```

Expected: prints `OK scale_sign signguard <float>`.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git add -f docs/superpowers/specs/2026-10-02-fl-hgt-poisoning-testbed-design.md docs/superpowers/plans/2026-10-02-fl-hgt-poisoning-testbed.md
git commit -m "docs(fl): document the FL-HGT poisoning testbed and run commands"
```

---

## Self-review notes

- Spec §3.1–3.12 map to Tasks 3, 4, 5, 11, 6, 7, 8, 9, 12, 10, 12 (outputs), 14 respectively; configs (§3.12 grids, tiers in §2) → Task 15; tests (§5) are embedded per task; limitations (§6) → CLAUDE.md note in Task 16.
- Test counts: 6 + 5 + 7 + 4 + 4 + 11 + 17 + 9 + 6 + 4 + 8 + 7 + 2 + 3 + 4 = 97 FL tests in the files above; Step 2 of Task 16 must report the actual number collected (update the expected count if a parametrisation differs).

