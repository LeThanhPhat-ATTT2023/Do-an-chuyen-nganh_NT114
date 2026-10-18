# FL-HGT Poisoning Testbed Implementation Plan (revision 2, 2026-10-18)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the federated-learning simulation of the GraphSLM IDS flow classifier: 20 clients × 20 rounds on an explicit class-share table, three models (HGT, MLP, 1D-CNN) behind one adapter, six comparison defenses plus the proposed `RAGuard`, offline detection replay and a resumable experiment matrix (P0/P1/P2).

**Architecture:** Package `src/graphslm_ids/offline/federated/`. Delta-space modules (`linalg`, `attacks`, `aggregators`, `defense`, `server`, `offline_eval`) work on `(N, D)` float32 tensors plus `ParamLayout` and never import model or graph code. Model-specific code sits behind a `ModelAdapter` (`models.py` for MLP/CNN, `hgt_adapter.py` for HGT), so `local_trainer.py` and `simulation.py` do not depend on the architecture. The HGT adapter reuses building blocks of `train_hgt_flow_classifier.py` without modifying it.

**Tech Stack:** Python 3.14, PyTorch 2.14 (CPU locally, CUDA on the L40S), NumPy, SciPy (`norm.ppf`), scikit-learn (`MeanShift`), PyYAML, psutil, matplotlib (heatmap only), pytest.

**Spec:** `docs/superpowers/specs/2026-10-02-fl-hgt-poisoning-testbed-design.md` (updated 2026-10-15). Task numbers T7–T16 below are the spec's §7 order.

## Status

- **Done (tasks 1–6 of revision 1, commits `7f73fa7` … `e80d122`):** `config.py`, `records.py`, `linalg.py`, `partition.py` (IID/Dirichlet), `client_graph.py`, `params.py`, `attacks.py`, `tests/federated/conftest.py` (tiny v3 fixture). 158 tests pass in `tests/federated`.
- **Revision 1 of this plan** (commit `539fc79`) carries full code sketches for aggregators, server, metrics, local trainer, simulation, CLI and matrix. They were never executed and target the old design (screeners, relation-aware wrapper, 10 Dirichlet clients). Read one with `git show 539fc79:docs/superpowers/plans/2026-10-02-fl-hgt-poisoning-testbed.md` when a task below says "reference sketch"; adapt it to the interfaces here, do not copy what this plan removed.

## Conventions (read once)

- Work in the repo root `D:\DACN\Project\Do-an-chuyen-nganh_NT114` on branch `feat/fl-poisoning-testbed` (no worktree).
- Interpreter `D:/v/nt114/Scripts/python.exe` (written `PY` below; works from Git Bash and PowerShell). `tests/conftest.py` puts `src/` on `sys.path`; anything run outside pytest needs `PYTHONPATH=src` (scripts under `scripts/fl/` insert it themselves).
- FL tests: `PY -m pytest tests/federated -q`. Whole repo: `PY -m pytest tests -q --ignore=tests/baselines` (the GNN4ID baseline tests need `torch_geometric`, which this venv does not have). Baseline before T7: 478 passed, 3 skipped.
- `docs/superpowers/` is gitignored but tracked: add files there with `git add -f`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Global Constraints

- Never modify `src/graphslm_ids/offline/training/train_hgt_flow_classifier.py`.
- `attacks.py`, `aggregators.py`, `defense.py`, `server.py`, `offline_eval.py` must not import model, graph or trainer code (only torch/numpy/scipy/sklearn and `linalg`, `params`, `records`).
- Default scale: `partition.num_clients = 20`, `federated.rounds = 20`, `federated.participation = 1.0`.
- `defense.assumed_malicious_ratio = 0.2`, fixed for every experiment and never derived from the attack config.
- Models must have float-only `state_dict` entries: no `BatchNorm` (use `GroupNorm` or nothing).
- Client RNG seed for round `r`, client `k`: `seed + 100003·r + 1009·k`.
- Headline metric is the **final-round** global model; best-val is reported alongside.
- Every config typo or unknown name (model, attack, defense, mode, monitor, placement, partition scheme) raises `ValueError` before any training.
- All tests run on CPU under `tests/federated/`.

## Review Focus

1. A label that belongs to no family in the explicit table (e.g. a class added later) must be split evenly like Benign, never dropped and never crash → test in T10 (`test_class_outside_every_family_is_split_evenly`).
2. A CNN flow with fewer than `K` packets, or none, must be zero-padded, not shifted or crash → test in T11 (`test_flow_packet_csr_and_byte_matrix`).
3. A round in which the defense rejects every client must leave `W_t` and `prev_aggregate` unchanged and report `applied = False` → test in T7 (`test_server_does_not_apply_when_every_client_is_rejected`).
4. Saving deltas as float16 must not turn large finite attack deltas into `inf` (the offline replay would then mark them invalid and disagree with the live run) → test in T14 (`test_float16_overflow_falls_back_to_float32`).
5. Resuming a matrix after a crashed cell (`error.txt`, no `fl_summary.json`) must rerun that cell and clear the stale error → test in T15 (`test_failed_cell_is_rerun_on_resume`).

## File map (after this plan)

| File | Responsibility | Task |
|---|---|---|
| `federated/config.py` | defaults (revised), overrides, `unknown_keys` | T7, T8, T10, T11, T14 |
| `federated/aggregators.py` | decision constants, `AggResult`, `Aggregator`, `weighted_mean`, FedAvg, Median, TrimmedMean, MultiKrum, SignGuard, DnC, `build_aggregator` | T7, T12 |
| `federated/server.py` | `ClientMessage`, `ClientDecision`, `RoundReport`, `FLServer` | T7 |
| `federated/metrics.py` | confusion-based IDS metrics, detection counts, convergence | T7 |
| `federated/context.py` | `ExperimentContext`, `load_context`, `data_key`, `federated_flow_stats` | T8 |
| `federated/models.py` | `ModelAdapter`, `StepOutput`, `build_adapter`, `rebuild_model`, `FlowMLP`, `ByteCNN1D`, MLP/CNN adapters, byte-matrix helpers | T8, T11 |
| `federated/hgt_adapter.py` | `HGTAdapter` (sampler, GCL, EACS, drop-edge) | T8 |
| `federated/local_trainer.py` | `ClientState`, `LocalUpdate`, `LocalTrainer`, `local_class_weights`, `round_lr` | T8 |
| `federated/simulation.py` | `run_experiment` (fl / centralized / local_only), selection helpers, evaluator, checkpoints | T9, T10, T13, T14 |
| `federated/cli.py` | one experiment from the command line | T9 |
| `federated/partition.py` | + explicit class-share table | T10 |
| `federated/defense.py` | `RAGuard`, `build_defense`, pure helpers | T13 |
| `federated/linalg.py` | + `weighted_trimmed_mean` | T13 |
| `federated/params.py` | + `ParamLayout.from_groups`, `groups_json` | T14 |
| `federated/offline_eval.py` | delta store format, `replay`, CLI | T14 |
| `federated/matrix.py` | `expand_grid`, `run_grid`, `summarize`, writers | T15 |
| `scripts/fl/export_partition_table.py` | client × class CSV + heatmap | T10 |
| `scripts/fl/run_matrix.py`, `scripts/fl/summarize_matrix.py` | thin CLIs over `matrix.py` | T15 |
| `configs/fl/*.yaml` | base config + pilot/P0/P1/P2 grids | T15 |

`context.py` and `hgt_adapter.py` are additions to the spec's file list (split out so `simulation.py` and `models.py` stay focused); T16 records them in the spec.

---

### Task 7: Revised config, minimal server, FedAvg and metrics

**Files:**
- Modify: `src/graphslm_ids/offline/federated/config.py`
- Create: `src/graphslm_ids/offline/federated/aggregators.py`, `src/graphslm_ids/offline/federated/server.py`, `src/graphslm_ids/offline/federated/metrics.py`
- Test: `tests/federated/test_config_records.py` (modify), `tests/federated/test_aggregators.py`, `tests/federated/test_server.py`, `tests/federated/test_metrics.py`

**Interfaces:**
- Consumes: `linalg` helpers; `records.to_jsonable`; `params.ParamLayout` (type only).
- Produces:
  - `aggregators.ACCEPT = "accept"`, `DOWNWEIGHT = "downweight"`, `REJECT = "reject"`, `INVALID = "invalid"`.
  - `@dataclass AggResult(delta: Tensor[D], selected: BoolTensor[N] | None = None, decisions: list[str] | None = None, trust: np.ndarray | None = None, weights: np.ndarray | None = None, scores: dict = {}, info: dict = {})`.
  - `class Aggregator: name: str; aggregate(self, deltas: Tensor[N, D], weights: Tensor[N], ctx: dict | None = None) -> AggResult`. `ctx` keys: `client_ids: list[int]`, `round_idx: int`, `layout: ParamLayout | None`, `prev_aggregate: Tensor[D] | None`.
  - `weighted_mean(deltas, weights) -> Tensor[D]` (`ValueError` when weights sum to ≤ 0); `FedAvg`; `build_aggregator(cfg: dict, *, assumed_malicious_ratio: float = 0.2, seed: int = 0) -> Aggregator` (registry; T12 adds rules).
  - `server.ClientMessage(client_id: int, round_idx: int, delta: Tensor[D], num_samples: int)`; `ClientDecision(client_id, decision, trust: float, weight: float, norm: float | None, reason: str = "")`; `RoundReport(round_idx, decisions: list[ClientDecision], applied: bool, aggregate_norm: float, seconds: float, info: dict, scores: dict)` with `decision_map() -> dict[int, str]`.
  - `FLServer(*, global_vec, aggregator, layout=None, weighting="samples", server_lr=1.0, device="cpu")`; attributes `global_vec`, `round_idx` (starts at 1), `prev_aggregate`, `last_inputs: tuple[list[int], Tensor[N, D], Tensor[N]] | None` (valid messages of the last step); `step(messages) -> RoundReport`.
  - `metrics.metrics_from_confusion(cm, label_names, families=None, benign="Benign") -> dict`, `evaluate_predictions(flow_ids, logits, labels, label_names, *, clean_labels=None, families=None) -> dict` (keys of `metrics_from_confusion` plus `loss`, `confusion`, optional `clean`), `detection_counts(decisions: dict[int, str], malicious: set[int]) -> dict`, `merge_counts`, `detection_rates(counts) -> dict` (`tpr`, `fpr`, `precision`, `malicious_acceptance_rate`), `convergence_round(history, frac=0.95)`, `compact_metrics(metrics) -> dict`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/federated/test_config_records.py`, and update any existing test there that names `defense.screen`, `defense.clip` or `defense.relation_aware` (those keys are removed):

```python
def test_defaults_follow_the_revised_spec():
    cfg = load_fl_config(None)
    assert cfg["federated"]["rounds"] == 20 and cfg["partition"]["num_clients"] == 20
    assert cfg["federated"]["participation"] == 1.0
    assert set(cfg["defense"]) == {"aggregator", "assumed_malicious_ratio", "weighting"}
    assert cfg["defense"]["aggregator"] == {"name": "fedavg", "params": {}}
    assert cfg["defense"]["assumed_malicious_ratio"] == 0.2
    assert unknown_keys(apply_overrides(cfg, {"defense.aggregator.params.anything": 1})) == []
    assert unknown_keys({"defense": {"screen": {"name": "x"}}}) == ["defense.screen"]
```

`tests/federated/test_aggregators.py`:

```python
import pytest
import torch

from graphslm_ids.offline.federated.aggregators import FedAvg, build_aggregator, weighted_mean


def test_fedavg_matches_weighted_mean():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert torch.allclose(FedAvg().aggregate(x, torch.tensor([1.0, 3.0])).delta, torch.tensor([2.5, 3.5]))


def test_weighted_mean_rejects_zero_total():
    with pytest.raises(ValueError):
        weighted_mean(torch.ones(2, 3), torch.zeros(2))


def test_build_aggregator_fedavg_and_errors():
    assert isinstance(build_aggregator({"name": "fedavg"}), FedAvg)
    with pytest.raises(ValueError):
        build_aggregator({"name": "nope"})
    with pytest.raises(ValueError):
        build_aggregator({"name": "fedavg", "params": {"bogus": 1}})
```

`tests/federated/test_server.py`:

```python
import numpy as np
import pytest
import torch

from graphslm_ids.offline.federated.aggregators import ACCEPT, REJECT, AggResult, Aggregator, FedAvg, weighted_mean
from graphslm_ids.offline.federated.server import ClientMessage, FLServer


class _Scripted(Aggregator):
    """Fixed decisions (or a selected mask); records the ctx it receives."""

    name = "scripted"

    def __init__(self, decisions=None, selected=None):
        self.decisions, self.selected, self.seen_ctx = decisions, selected, None

    def aggregate(self, deltas, weights, ctx=None):
        self.seen_ctx = ctx
        if self.decisions is not None:
            keep = torch.tensor([d != REJECT for d in self.decisions])
            delta = weighted_mean(deltas[keep], weights[keep]) if keep.any() else torch.zeros(deltas.shape[1])
            trust = np.array([0.0 if d == REJECT else 1.0 for d in self.decisions])
            return AggResult(delta, decisions=list(self.decisions), trust=trust, weights=trust * weights.numpy())
        mask = torch.as_tensor(self.selected)
        return AggResult(weighted_mean(deltas[mask], weights[mask]), selected=mask)


def _msg(cid, vec, r=1, n=1):
    return ClientMessage(cid, r, torch.tensor(vec, dtype=torch.float32), n)


def test_server_applies_weighted_fedavg_and_advances_round():
    server = FLServer(global_vec=torch.zeros(3), aggregator=FedAvg())
    report = server.step([_msg(0, [1.0, 0.0, 0.0], n=1), _msg(1, [0.0, 2.0, 0.0], n=3)])
    assert report.applied and server.round_idx == 2
    assert torch.allclose(server.global_vec, torch.tensor([0.25, 1.5, 0.0]))
    assert torch.allclose(server.prev_aggregate, torch.tensor([0.25, 1.5, 0.0]))
    assert report.decision_map() == {0: "accept", 1: "accept"}


def test_server_rejects_invalid_messages():
    server = FLServer(global_vec=torch.zeros(2), aggregator=FedAvg())
    report = server.step(
        [_msg(0, [float("nan"), 0.0]), _msg(1, [1.0, 1.0], r=5), _msg(2, [1.0, 1.0, 1.0]), _msg(3, [1.0, 1.0])]
    )
    assert report.decision_map() == {0: "invalid", 1: "invalid", 2: "invalid", 3: "accept"}
    assert torch.allclose(server.global_vec, torch.ones(2))
    ids, deltas, samples = server.last_inputs
    assert ids == [3] and tuple(deltas.shape) == (1, 2) and samples.tolist() == [1.0]


def test_server_keeps_global_when_nothing_is_valid():
    server = FLServer(global_vec=torch.ones(2), aggregator=FedAvg())
    report = server.step([_msg(0, [float("inf"), 0.0])])
    assert not report.applied and server.round_idx == 2
    assert torch.allclose(server.global_vec, torch.ones(2)) and server.prev_aggregate is None


def test_server_uses_aggregator_decisions_and_server_lr():
    agg = _Scripted(decisions=[REJECT, ACCEPT])
    server = FLServer(global_vec=torch.zeros(2), aggregator=agg, server_lr=0.5)
    report = server.step([_msg(0, [-9.0, -9.0]), _msg(1, [2.0, 2.0])])
    assert report.decision_map() == {0: "reject", 1: "accept"}
    assert torch.allclose(server.global_vec, torch.tensor([1.0, 1.0]))
    assert agg.seen_ctx["client_ids"] == [0, 1] and agg.seen_ctx["round_idx"] == 1
    assert "prev_aggregate" in agg.seen_ctx and "layout" in agg.seen_ctx


def test_server_marks_unselected_clients_rejected():
    server = FLServer(global_vec=torch.zeros(2), aggregator=_Scripted(selected=[False, True]))
    report = server.step([_msg(0, [5.0, 5.0]), _msg(1, [1.0, 1.0])])
    assert report.decision_map() == {0: "reject", 1: "accept"}
    assert torch.allclose(server.global_vec, torch.ones(2))


def test_server_does_not_apply_when_every_client_is_rejected():
    server = FLServer(global_vec=torch.ones(2), aggregator=_Scripted(decisions=[REJECT, REJECT]))
    report = server.step([_msg(0, [5.0, 5.0]), _msg(1, [1.0, 1.0])])
    assert not report.applied and server.round_idx == 2
    assert torch.allclose(server.global_vec, torch.ones(2)) and server.prev_aggregate is None


def test_server_weighting_modes():
    server = FLServer(global_vec=torch.zeros(1), aggregator=FedAvg(), weighting="uniform")
    server.step([_msg(0, [1.0], n=1), _msg(1, [3.0], n=9)])
    assert torch.allclose(server.global_vec, torch.tensor([2.0]))
    with pytest.raises(ValueError):
        FLServer(global_vec=torch.zeros(1), aggregator=FedAvg(), weighting="bogus")
```

`tests/federated/test_metrics.py`: the four tests of the revision-1 sketch (Task 10: `test_metrics_from_confusion_rates_and_families`, `test_macro_f1_ignores_classes_without_support`, `test_detection_counts_and_rates`, `test_convergence_round_and_compact`) plus:

```python
def test_evaluate_predictions_noisy_and_clean():
    labels = np.array([0, 1, 2, 1])
    clean = np.array([0, 0, 2, 1])
    gids = np.array([3, 1, 0, 2])  # adapters may return flows in any order
    logits = torch.tensor([[0, 5, 0], [0, 5, 0], [5, 0, 0], [0, 0, 5]], dtype=torch.float32)
    m = evaluate_predictions(gids, logits, labels, NAMES, clean_labels=clean)
    assert m["count"] == 4 and m["accuracy"] == pytest.approx(1.0)
    assert m["clean"]["accuracy"] == pytest.approx(0.75)
    assert m["loss"] > 0 and m["confusion"][1][1] == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PY -m pytest tests/federated/test_config_records.py tests/federated/test_aggregators.py tests/federated/test_server.py tests/federated/test_metrics.py -q`
Expected: FAIL — `ModuleNotFoundError` for `aggregators`/`server`/`metrics`, and the config defaults assertion.

- [ ] **Step 3: Revise `DEFAULT_FL_CONFIG` in `config.py`**

`federated.rounds: 20`, `partition.num_clients: 20`; replace the `defense` section with `{"aggregator": {"name": "fedavg", "params": {}}, "assumed_malicious_ratio": 0.2, "weighting": "samples"}`; drop `"defense.screen.params"` from `OPEN_SUBTREES`. Leave every other key as is (T8/T10/T11/T14 add theirs).

- [ ] **Step 4: Implement `aggregators.py` (constants, `AggResult`, `Aggregator`, `weighted_mean`, `FedAvg`, `build_aggregator`)**

`build_aggregator` reads `cfg["name"]` (lower-cased) from a registry dict and passes `cfg["params"]` as kwargs; a `TypeError` from the constructor becomes `ValueError`. Reference sketch: revision-1 Task 7 (drop `clip_deltas`, `relation_aware_aggregate`, `MSGuard`, `coordinatewise`).

- [ ] **Step 5: Implement `server.py`**

`step`: validate each message (`round_idx == self.round_idx`, 1-D with `D` entries, all finite; otherwise `INVALID` with reason) → stack valid deltas, base weights = `num_samples` (or ones for `uniform`) → `aggregator.aggregate(deltas, weights, ctx)` with the four ctx keys → resolve per-client decisions: `result.decisions` if given (trust/weights from the result), else `selected` (unselected ⇒ `REJECT`), else all `ACCEPT` with trust 1.0 → apply `global_vec += server_lr * delta` and set `prev_aggregate` **only if at least one decision is not `REJECT`** → `round_idx += 1` always. Store `last_inputs` before aggregating. `RoundReport.info`/`scores` come from `to_jsonable(result.info/scores)`. Reference sketch: revision-1 Task 9 (drop screener, clip, relation-aware).

- [ ] **Step 6: Implement `metrics.py`**

Reference sketch: revision-1 Task 10 for everything except `confusion_from_loader`, which is replaced by `evaluate_predictions`: `y = labels[flow_ids]`, `pred = logits.argmax(1)`, `loss = cross_entropy(logits, y)` (mean), `confusion = cm.tolist()`, and the same for `clean_labels[flow_ids]` under `"clean"`. No trainer import.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `PY -m pytest tests/federated -q`
Expected: all pass (158 existing + the new ones).

- [ ] **Step 8: Commit**

```bash
git add src/graphslm_ids/offline/federated/{config,aggregators,server,metrics}.py tests/federated/test_{config_records,aggregators,server,metrics}.py
git commit -m "feat(fl): revised defaults, FedAvg server with integrity checks, IDS/poisoning metrics"
```

---

### Task 8: Data context, `ModelAdapter`, `HGTAdapter`, generic local trainer

**Files:**
- Modify: `src/graphslm_ids/offline/federated/config.py` (add `model.name`)
- Modify: `tests/federated/conftest.py` (append `smoke_cfg`)
- Create: `src/graphslm_ids/offline/federated/context.py`, `models.py`, `hgt_adapter.py`, `local_trainer.py`
- Test: `tests/federated/test_context.py`, `tests/federated/test_local_trainer.py`

**Interfaces:**
- Consumes: `client_graph.packet_owner`, `build_client_artifact`; `params.ParamLayout`; trainer helpers `to_torch_batch`, `make_neighbor_loader`, `_drop_edges`, `_compute_train_loss`, `gcl_auxiliary_loss`, `backend_splits`, `label_name_mapping`, `load_class_technique_map`, `_set_tew_state`; `noise_consensus.{EACSController, build_eacs_controller, neighbor_consensus, soft_target_focal_loss}`; `neighbor_sampling.{HeteroNeighborSampler, InMemoryNeighborBackend}`; `hetero_graph_artifact.load_v3_artifact`; `models.hgt.HeteroGraphTransformer`.
- Produces:
  - `context.ExperimentContext` dataclass: `data_key: str, artifact, backend, labels: int64[F], train_ids, val_ids, test_ids, label_names: dict[int, str], num_classes: int, clean_labels: np.ndarray | None, packet_owner: int64[P], flow_order: np.ndarray | None, client_cache: dict[int, Any] = {}, cache_tag: str | None = None`.
  - `context.data_key(cfg) -> str` (sha1 of `cfg["data"]`), `load_context(cfg) -> ExperimentContext`, `federated_flow_stats(flow_x: np.ndarray, client_flow_ids: list[np.ndarray]) -> {"mean": list[float], "std": list[float]}`.
  - `models.StepOutput(loss: Tensor, logits: Tensor, labels: Tensor)` (logits detached).
  - `models.ModelAdapter` with `name: str` and methods `build_model() -> nn.Module`, `client_data(client_id: int, flow_ids: np.ndarray, seed: int) -> Any`, `client_eacs() -> EACSController | None` (base: `None`), `train_batches(data, flow_ids: np.ndarray, labels: np.ndarray, *, seed: int) -> Iterable`, `forward_loss(model, batch, *, class_weights: Tensor | None, eacs, round_idx: int) -> StepOutput | None`, `predict(model, flow_ids: np.ndarray) -> tuple[np.ndarray, Tensor]` (global ids, float32 logits `(n, C)` on CPU), `checkpoint_meta() -> dict` (always includes `model_name`).
  - `models.build_adapter(cfg, ctx, flow_stats, device) -> ModelAdapter` (`model.name` ∈ {`hgt`}; T11 adds `mlp`, `cnn`; unknown ⇒ `ValueError`); `models.rebuild_model(ckpt: dict) -> nn.Module` (HGT here, MLP/CNN in T11).
  - `hgt_adapter.HGTAdapter(cfg, ctx, flow_stats, device)`; attribute `gcl_enabled: bool`.
  - `local_trainer.ClientState(client_id, flow_ids, flow_labels, data, class_weights=None, eacs=None, optimizer_state=None, rounds_trained=0)`, `LocalUpdate(client_id, delta: Tensor[D], num_samples, steps, seconds, metrics: dict)`, `LocalTrainer(model, layout, adapter, device, client_cfg)` with `train(state, global_vec, round_idx, lr, seed) -> LocalUpdate`, `local_class_weights(labels, num_classes, method="cb", beta=0.985, cap=4.0) -> Tensor[C]`, `round_lr(client_cfg, round_idx, total_rounds) -> float`.

- [ ] **Step 1: Add the `smoke_cfg` fixture and write the failing tests**

Append to `tests/federated/conftest.py` (dotted keys only, so later tasks' new defaults under `model`/`partition` survive):

```python
@pytest.fixture
def smoke_cfg(tiny_paths, tmp_path):
    """Factory: CPU smoke config on the tiny artifact writing to tmp_path/<name>."""
    from graphslm_ids.offline.federated.config import apply_overrides, load_fl_config

    def make(name: str = "run", **overrides):
        base = {
            "data.graph_npz": str(tiny_paths["graph_npz"]),
            "data.graph_meta_json": str(tiny_paths["graph_meta_json"]),
            "data.splits_json": str(tiny_paths["splits_json"]),
            "data.clean_eval_labels": str(tiny_paths["clean_eval_labels"]),
            "data.packet_dtype": "float32",
            "model.hidden_dim": 8,
            "model.num_layers": 2,
            "model.num_heads": 2,
            "model.dropout": 0.0,
            "sampler.hops": 2,
            "partition.scheme": "iid",
            "partition.num_clients": 3,
            "partition.min_flows_per_client": 1,
            "partition.max_flows_per_client": None,
            "partition.seed": 0,
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

`tests/federated/test_context.py`:

```python
import json

import numpy as np
import pytest

from graphslm_ids.offline.federated.config import apply_overrides
from graphslm_ids.offline.federated.context import data_key, federated_flow_stats, load_context


def test_load_context_reads_splits_labels_and_clean_key(smoke_cfg, tiny_paths):
    ctx = load_context(smoke_cfg())
    split = json.loads(tiny_paths["splits_json"].read_text(encoding="utf-8"))["random"]
    assert sorted(ctx.train_ids.tolist()) == split["train"] and sorted(ctx.val_ids.tolist()) == split["val"]
    assert ctx.label_names[1] == "XSS" and ctx.num_classes == 3
    assert ctx.clean_labels.shape == (48,) and ctx.packet_owner.shape == (96,)
    assert ctx.flow_order is None


def test_load_context_rejects_wrong_clean_length(smoke_cfg, tmp_path):
    bad = tmp_path / "bad.npy"
    np.save(bad, np.zeros(5, dtype=np.int64))
    with pytest.raises(ValueError):
        load_context(smoke_cfg(**{"data.clean_eval_labels": str(bad)}))


def test_data_key_tracks_the_data_section(smoke_cfg):
    cfg = smoke_cfg()
    assert data_key(cfg) == data_key(smoke_cfg("other"))
    assert data_key(cfg) != data_key(apply_overrides(cfg, {"data.split_protocol": "temporal"}))


def test_federated_flow_stats_matches_pooled_moments():
    x = np.random.default_rng(0).normal(size=(10, 3)).astype(np.float32)
    x[:, 0] = 6.0
    s = federated_flow_stats(x, [np.array([0, 1, 2]), np.array([5, 7])])
    rows = x[[0, 1, 2, 5, 7]].astype(np.float64)
    assert np.allclose(s["mean"], rows.mean(0))
    assert np.allclose(s["std"][1:], rows.std(0)[1:], rtol=1e-6)
    assert s["std"][0] == pytest.approx(1e-6)
```

`tests/federated/test_local_trainer.py`:

```python
import numpy as np
import pytest
import torch

from graphslm_ids.offline.federated.context import federated_flow_stats, load_context
from graphslm_ids.offline.federated.local_trainer import ClientState, LocalTrainer, local_class_weights, round_lr
from graphslm_ids.offline.federated.models import build_adapter, rebuild_model
from graphslm_ids.offline.federated.params import ParamLayout

EVIDENCE = "rel:packet__evidence_injection__technique"


def _setup(smoke_cfg, flow_ids, **overrides):
    cfg = smoke_cfg("lt", **{"client.drop_edge_prob": 0.0, **overrides})
    ctx = load_context(cfg)
    flow_ids = np.asarray(flow_ids, dtype=np.int64)
    stats = federated_flow_stats(ctx.artifact.node_features["flow"], [flow_ids])
    adapter = build_adapter(cfg, ctx, stats, torch.device("cpu"))
    torch.manual_seed(0)
    model = adapter.build_model()
    layout = ParamLayout.from_model(model)
    labels = ctx.labels[flow_ids]
    state = ClientState(0, flow_ids, labels, adapter.client_data(0, flow_ids, seed=0), local_class_weights(labels, ctx.num_classes))
    return LocalTrainer(model, layout, adapter, "cpu", cfg["client"]), layout, state, adapter, ctx


def test_local_class_weights_ignore_absent_classes():
    w = local_class_weights(np.array([0, 0, 0, 1]), 3, method="cb", beta=0.9, cap=10.0)
    assert w[2].item() == 1.0 and float(w[:2].mean()) == pytest.approx(1.0) and float(w[1]) > float(w[0])
    with pytest.raises(ValueError):
        local_class_weights(np.array([0]), 2, method="bogus")


def test_round_lr_schedules():
    cfg = {"lr": 1e-2, "lr_min": 1e-4, "lr_schedule": "cosine_rounds"}
    assert round_lr(cfg, 1, 10) == pytest.approx(1e-2)
    assert round_lr(cfg, 10, 10) < round_lr(cfg, 5, 10) < round_lr(cfg, 1, 10)
    assert round_lr({"lr": 0.1, "lr_schedule": "constant"}, 7, 10) == 0.1
    with pytest.raises(ValueError):
        round_lr({"lr": 0.1, "lr_schedule": "bogus"}, 1, 2)


def test_local_trainer_returns_finite_delta(smoke_cfg):
    trainer, layout, state, _, _ = _setup(smoke_cfg, np.arange(30))
    update = trainer.train(state, layout.flatten(trainer.model.state_dict()), round_idx=1, lr=1e-2, seed=0)
    assert update.delta.shape == (layout.numel,) and bool(torch.isfinite(update.delta).all())
    assert update.steps > 0 and float(update.delta.abs().sum()) > 0
    assert update.num_samples == 30 and state.rounds_trained == 1


def test_relation_absent_from_client_has_exactly_zero_delta(smoke_cfg, tiny_artifact):
    y = np.asarray(tiny_artifact.flow_y)
    trainer, layout, state, _, _ = _setup(smoke_cfg, np.flatnonzero(y == 0))
    update = trainer.train(state, layout.flatten(trainer.model.state_dict()), 1, 1e-2, 0)
    assert int(torch.count_nonzero(update.delta[layout.group_coords([EVIDENCE])])) == 0
    trainer2, layout2, state2, _, _ = _setup(smoke_cfg, np.flatnonzero(y == 1))
    update2 = trainer2.train(state2, layout2.flatten(trainer2.model.state_dict()), 1, 1e-2, 0)
    assert int(torch.count_nonzero(update2.delta[layout2.group_coords([EVIDENCE])])) > 0


def test_eacs_soft_target_path_runs_after_warmup(smoke_cfg):
    trainer, layout, state, adapter, _ = _setup(smoke_cfg, np.arange(30))
    state.eacs = adapter.client_eacs()
    update = trainer.train(state, layout.flatten(trainer.model.state_dict()), 3, 1e-2, 0)
    assert bool(torch.isfinite(update.delta).all()) and state.eacs.epoch_stats()["suspects_seen"] > 0


def test_gcl_path_runs(smoke_cfg):
    trainer, layout, state, adapter, _ = _setup(smoke_cfg, np.arange(30))
    assert adapter.gcl_enabled
    assert bool(torch.isfinite(trainer.train(state, layout.flatten(trainer.model.state_dict()), 1, 1e-2, 0).delta).all())


def test_persisted_optimizer_state_lives_on_cpu(smoke_cfg):
    trainer, layout, state, _, _ = _setup(smoke_cfg, np.arange(30), **{"client.optimizer_state": "persist"})
    vec = layout.flatten(trainer.model.state_dict())
    update = trainer.train(state, vec, 1, 1e-2, 0)
    for per_param in state.optimizer_state["state"].values():
        for value in per_param.values():
            if isinstance(value, torch.Tensor):
                assert value.device.type == "cpu"
    trainer.train(state, vec + update.delta, 2, 1e-2, 1)
    assert state.rounds_trained == 2


def test_trainer_rejects_bad_config(smoke_cfg):
    with pytest.raises(ValueError):
        _setup(smoke_cfg, np.arange(5), **{"client.optimizer_state": "bogus"})
    with pytest.raises(ValueError):
        _setup(smoke_cfg, np.arange(5), **{"client.loss_type": "bogus"})
    with pytest.raises(ValueError):
        _setup(smoke_cfg, np.arange(5), **{"model.name": "bogus"})


def test_hgt_predict_and_checkpoint_meta(smoke_cfg):
    trainer, _, _, adapter, ctx = _setup(smoke_cfg, np.arange(30))
    gids, logits = adapter.predict(trainer.model, ctx.val_ids)
    assert sorted(gids.tolist()) == sorted(ctx.val_ids.tolist())
    assert tuple(logits.shape) == (len(ctx.val_ids), 3) and bool(torch.isfinite(logits).all())
    meta = adapter.checkpoint_meta()
    assert meta["model_name"] == "hgt" and {"node_input_dims", "edge_types", "num_tactics"} <= set(meta)
    ckpt = {**meta, "model_state_dict": trainer.model.state_dict(), "config": smoke_cfg(), "num_classes": 3}
    rebuild_model(ckpt).load_state_dict(ckpt["model_state_dict"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PY -m pytest tests/federated/test_context.py tests/federated/test_local_trainer.py -q`
Expected: FAIL — `ModuleNotFoundError: ... federated.context`.

- [ ] **Step 3: Add `model.name: "hgt"` to `DEFAULT_FL_CONFIG`**

- [ ] **Step 4: Implement `context.py`**

`load_context` is the revision-1 `load_context` (Task 12 sketch) plus `flow_order = np.asarray(metadata["flow_start_ts"], dtype=np.float64)` when the artifact metadata has that key (it does not today; spec §3.1 falls back to flow id), otherwise `None`. `federated_flow_stats`: pooled `n, Σx, Σx²` in float64, `std = max(sqrt(max(E[x²] − mean², 0)), 1e-6)`.

- [ ] **Step 5: Implement `models.py` (base pieces) and `hgt_adapter.py`**

`HGTAdapter` holds what revision-1 `LocalTrainer` and `simulation` did for HGT: `build_model` (= revision-1 `build_model`), `client_data` → `HGTClientData(backend, sampler)` from `build_client_artifact(ctx.artifact, flow_ids, owner=ctx.packet_owner)` and `make_sampler(..., flow_stats, seed)`; `client_eacs` → a fresh `EACSController` sharing the masks of one base controller built once in `__init__` (revision-1 `build_base_eacs` + per-client copy in `build_clients`); `train_batches` resets `data.sampler.rng = np.random.default_rng(seed)` and returns `make_neighbor_loader(..., shuffle=True, class_labels=labels)`; `forward_loss` = revision-1 `LocalTrainer._step` (drop-edge, GCL via `encode` + `classifier`, EACS soft targets, `_compute_train_loss` otherwise); `predict` builds a full-graph sampler with the run's `flow_stats` and a fixed seed, iterates `make_neighbor_loader(flow_ids, shuffle=False)` under `inference_mode` and concatenates `batch.seed_flow_ids` and `logits[seed_mask]`; `checkpoint_meta` returns `model_name`, `node_input_dims`, `edge_types`, `num_tactics`, `label_names`, `flow_feature_stats`. Call `_set_tew_state(False)` in `__init__`. `rebuild_model(ckpt)` constructs `HeteroGraphTransformer` from those keys and `ckpt["config"]["model"]`.

- [ ] **Step 6: Implement `local_trainer.py`**

Revision-1 Task 11 sketch with the HGT parts moved out: `train` loads `global_vec`, builds AdamW (restoring CPU-stored state when `optimizer_state == "persist"`), loops `local_epochs` × `adapter.train_batches(state.data, state.flow_ids, state.flow_labels, seed=seed + epoch)`, calls `adapter.forward_loss(..., class_weights=state.class_weights, eacs=state.eacs, round_idx=round_idx)`, skips `None` and non-finite losses, clips grads, stops at `max_local_steps`, returns `delta = flatten(trained) − global_vec`. Validate `optimizer_state ∈ {reset, persist}` and `loss_type ∈ {ce, focal, cb_focal}` in `__init__`.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `PY -m pytest tests/federated -q`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/graphslm_ids/offline/federated/{config,context,models,hgt_adapter,local_trainer}.py tests/federated/{conftest,test_context,test_local_trainer}.py
git commit -m "feat(fl): data context, ModelAdapter with HGT adapter, model-agnostic local trainer"
```

---

### Task 9: Simulation (`fl` / `centralized` / `local_only`) and CLI

**Files:**
- Create: `src/graphslm_ids/offline/federated/simulation.py`, `src/graphslm_ids/offline/federated/cli.py`
- Test: `tests/federated/test_simulation.py`, `tests/federated/test_cli.py`

**Interfaces:**
- Consumes: T7 (`FLServer`, `build_aggregator`, metrics), T8 (`load_context`, `data_key`, `federated_flow_stats`, `build_adapter`, `rebuild_model`, `LocalTrainer`, `ClientState`, `local_class_weights`, `round_lr`), `attacks.{build_attack, AttackContext}`, `partition.{make_partition, ClientPartition}`.
- Produces:
  - `VALID_MODES = ("fl", "centralized", "local_only")`, `VALID_MONITORS = ("val_macro_f1", "val_macro_f1_clean", "val_loss")`.
  - `select_participants(num_clients, fraction, seed, round_idx) -> list[int]`; `select_malicious(num_clients, ratio, seed) -> list[int]` (T10 adds `placement`, `groups` keywords).
  - `run_experiment(cfg, *, out_dir=None, context=None) -> dict` (summary; also written to `fl_summary.json`). Summary keys used later: `experiment`, `labels`, `mode`, `model`, `num_clients`, `client_sizes`, `malicious_clients`, `attack`, `aggregator`, `rounds`, `final.{val,test}`, `best.{val,test}`, `detection` (counts + rates), `convergence_round`, `history`, `cost`, and for `local_only` `per_client` + `aggregate`.
  - Run directory files: `config.resolved.yaml`, `partition.json`, `rounds.jsonl`, `fl_summary.json`, `global_final.pt`, `global_best.pt` (when `output.save_checkpoints`). Checkpoints = `adapter.checkpoint_meta()` + `model_state_dict`, `config`, `num_classes`, `label_names`, `epoch`, `val_metrics`, `test_metrics`, `fl`.
  - `cli.main(argv) -> int`, `cli.headline(summary) -> dict`.

- [ ] **Step 1: Write the failing tests**

`tests/federated/test_simulation.py` — the revision-1 Task 12 tests, changed as follows: drop `test_fl_smoke_relation_aware_multi_signal_min_max`; in `test_fl_smoke_sign_flip_trimmed_mean` keep the default FedAvg aggregator (rename `test_fl_smoke_sign_flip_fedavg`, remove the `defense.*` overrides) and replace the manual `HeteroGraphTransformer(...)` construction with:

```python
    ckpt = torch.load(out / "global_final.pt", map_location="cpu", weights_only=False)
    assert ckpt["model_name"] == "hgt" and ckpt["label_names"][1] == "XSS" and ckpt["flow_feature_stats"]["mean"]
    rebuild_model(ckpt).load_state_dict(ckpt["model_state_dict"])
    assert summary["model"] == "hgt"
```

In `test_fl_smoke_omniscient_alie_skips_attacker_training` drop the `defense.screen` override and the `summary["screen"]` assertion. In `test_context_reuse_and_validation` add:

```python
    with pytest.raises(ValueError):
        run_experiment(apply_overrides(cfg, {"model.name": "bogus"}))
```

and assert `ctx.client_cache` is non-empty after the first run (cache tag set by the simulation).

`tests/federated/test_cli.py`: revision-1 Task 13 tests unchanged.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PY -m pytest tests/federated/test_simulation.py tests/federated/test_cli.py -q`
Expected: FAIL — `ModuleNotFoundError: ... federated.simulation`.

- [ ] **Step 3: Implement `simulation.py`**

Reference sketch: revision-1 Task 12, with these changes:
- Validation before anything runs: mode, monitor, `unknown_keys(cfg)`, `model.name` (via `build_adapter`), context `data_key` match.
- `centralized` overrides: `client.optimizer_state: persist`, `attack.name: none`, `attack.malicious_ratio: 0`, `defense.aggregator: {name: fedavg, params: {}}`, `federated.participation: 1.0`. `local_only`: `client.optimizer_state: persist`.
- Partition: `make_partition(ctx.train_ids, ctx.labels, cfg["partition"])`; centralized wraps the union of client flows as one client (`scheme: "centralized"`). `partition.json` = `partition.to_json()` + `sizes` + `class_histogram`.
- Client data cache: `tag = f"{model.name}:{sha1 of all client flow ids}"`; if `ctx.cache_tag != tag`, clear `ctx.client_cache` and set the tag; `data = ctx.client_cache.get(cid)` else `adapter.client_data(cid, ids, seed + cid)` stored when `federated.cache_clients`.
- `flow_stats = federated_flow_stats(ctx.artifact.node_features["flow"], partition.client_flow_ids)` (round-0 moments), passed to `build_adapter`.
- Clients: `ClientState(cid, ids, labels, data, local_class_weights(...) if client.class_weight == "balanced" else None, adapter.client_eacs())`.
- Server: `FLServer(global_vec=init_vec, aggregator=build_aggregator(cfg["defense"]["aggregator"], assumed_malicious_ratio=..., seed=seed), layout=layout, weighting=..., server_lr=..., device=device)`.
- Evaluator: `adapter.predict(model, ids)` → `evaluate_predictions(gids, logits, ctx.labels, ctx.label_names, clean_labels=ctx.clean_labels, families=cfg["eval"]["families"])`.
- Round record keeps the revision-1 fields minus `screen_features`; add `"trust"` per decision (from `ClientDecision`) and `"agg_info"` = `report.info`.
- Summary adds `"model": cfg["model"]["name"]` and drops `"screen"`.

- [ ] **Step 4: Implement `cli.py`**

Revision-1 Task 13 sketch; `headline` reports `model` instead of `screen`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `PY -m pytest tests/federated -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/graphslm_ids/offline/federated/{simulation,cli}.py tests/federated/test_{simulation,cli}.py
git commit -m "feat(fl): experiment driver with fl, centralized and local_only modes plus CLI"
```

---

### Task 10: Explicit class-share partition, group attacker placement, partition table export

**Files:**
- Modify: `src/graphslm_ids/offline/federated/partition.py`, `config.py`, `simulation.py`
- Create: `scripts/fl/export_partition_table.py`
- Test: `tests/federated/test_partition_explicit.py`, `tests/federated/test_simulation.py` (append)

**Interfaces:**
- Consumes: `partition._stratified_quota(counts, k)` (existing largest-remainder quota for one client).
- Produces:
  - `ClientPartition` gains `counts: np.ndarray | None = None` (`(K, C)` int64) and `params: dict = {}`; `to_json` writes both (`counts` as nested list or `None`), `from_json` reads them with those defaults (old JSON still loads).
  - `group_members(groups: dict, num_clients: int) -> dict[str, list[int]]` (0-based, validated).
  - `explicit_fractions(label_names: dict[int, str], num_clients: int, groups: dict, families: dict[str, list[str]], dominance: float) -> np.ndarray` (`(K, C)` float64).
  - `explicit_counts(fractions: np.ndarray, class_counts: np.ndarray) -> np.ndarray` (`(K, C)` int64).
  - `partition_explicit(train_ids, labels, label_names, cfg: dict, order: np.ndarray | None = None) -> ClientPartition`.
  - `make_partition(train_ids, labels, cfg, *, label_names=None, order=None)`; scheme `explicit` requires `label_names`.
  - `simulation.select_malicious(num_clients, ratio, seed, *, placement="random", groups=None) -> list[int]`.
  - Config: `partition.scheme: "explicit"`, `partition.dominance: 0.8`, `partition.groups`, `partition.families`, `attack.placement: "random"`; `partition.groups` and `partition.families` are added to `OPEN_SUBTREES`.

Default config values (spec §3.1):

```yaml
partition:
  scheme: explicit
  num_clients: 20
  dominance: 0.8
  min_flows_per_client: 50
  groups:            # 1-based inclusive client ranges
    G1: {clients: [1, 5], family: ddos}
    G2: {clients: [6, 10], family: recon}
    G3: {clients: [11, 14], family: web}
    G4: {clients: [15, 16], family: other}
    G5: {clients: [17, 20], family: null}
  families:          # DEFAULT_FAMILIES with mirai + malware merged into "other"
    ddos: [DDoS-ACK_Fragmentation, DDoS-ICMP_Flood, DDoS-ICMP_Fragmentation, DDoS-PSHACK_Flood, DDoS-RSTFINFlood]
    recon: [Recon-HostDiscovery, Recon-OSScan, Recon-PingSweep, Recon-PortScan, VulnerabilityScan]
    web: [CommandInjection, XSS, SqlInjection, Uploading_Attack, BrowserHijacking]
    other: [Backdoor_Malware, Mirai-udpplain]
```

Algorithm (spec §3.1; not determined by the signatures):
- Fractions: class `c` in family `F` whose group has `n_F` clients: `p/n_F + (1−p)/K` inside the group, `(1−p)/K` outside. Any class in no family (Benign, a name no family lists, or a family no group uses): `1/K` for every client.
- Counts: `n[k, c] = floor(f[k, c] · N_c)`; hand the `N_c − Σ floor` leftovers one each to the clients with the largest fractional part, ties to the smaller client index (stable sort on `(−frac, k)`).
- Flow choice: train flows of class `c` sorted by `(order[id], id)` (or by `id` when `order is None`); cut into consecutive runs of lengths `n[0, c], n[1, c], …`; client `k` gets run `k`. Client arrays are stored sorted by id.
- Cap (`max_flows_per_client`): for a client with total `> cap`, `row' = _stratified_quota(row, cap)` and keep the **first** `row'[c]` flows of each of its runs (in time order). `counts` then holds the capped table.
- Validation (`ValueError`, before any flow is assigned): `0 ≤ dominance ≤ 1`; group ranges inside `1..K`, disjoint, covering all `K`; a group's `family` is `null` or a key of `families`; each family is used by at most one group; column sums of `f` equal 1 within `1e-9`; every client `≥ min_flows_per_client` after the cap.
- `params` records `dominance`, `groups`, `families`, `ordered_by: "flow_start_ts" | "flow_id"`, `max_flows_per_client`.

- [ ] **Step 1: Write the failing tests**

`tests/federated/test_partition_explicit.py`:

```python
import copy

import numpy as np
import pytest

from graphslm_ids.offline.federated.config import load_fl_config
from graphslm_ids.offline.federated.partition import (
    ClientPartition,
    explicit_counts,
    explicit_fractions,
    group_members,
    make_partition,
    partition_explicit,
)
from graphslm_ids.offline.federated.simulation import select_malicious

NAMES = [
    "Benign", "DDoS-ACK_Fragmentation", "DDoS-ICMP_Flood", "DDoS-ICMP_Fragmentation", "DDoS-PSHACK_Flood",
    "DDoS-RSTFINFlood", "Recon-HostDiscovery", "Recon-OSScan", "Recon-PingSweep", "Recon-PortScan",
    "VulnerabilityScan", "CommandInjection", "XSS", "SqlInjection", "Uploading_Attack", "BrowserHijacking",
    "Backdoor_Malware", "Mirai-udpplain",
]
LABEL_NAMES = dict(enumerate(NAMES))
C = {n: i for i, n in enumerate(NAMES)}


def _cfg(**kw):
    cfg = copy.deepcopy(load_fl_config(None)["partition"])
    cfg.update(kw)
    return cfg


def _data(per_class=400, extra=()):
    labels = np.repeat(np.arange(18 + len(extra)), per_class)
    return np.arange(labels.size, dtype=np.int64), labels


def test_default_partition_config_is_the_explicit_table():
    cfg = _cfg()
    assert cfg["scheme"] == "explicit" and cfg["dominance"] == 0.8 and cfg["num_clients"] == 20
    assert group_members(cfg["groups"], 20) == {
        "G1": [0, 1, 2, 3, 4], "G2": [5, 6, 7, 8, 9], "G3": [10, 11, 12, 13], "G4": [14, 15], "G5": [16, 17, 18, 19]
    }


@pytest.mark.parametrize(
    "p, ddos_in, web_in, other_in, out",
    [(0.8, 0.17, 0.21, 0.41, 0.01), (0.5, 0.125, 0.15, 0.275, 0.025), (0.0, 0.05, 0.05, 0.05, 0.05)],
)
def test_fraction_table_matches_spec(p, ddos_in, web_in, other_in, out):
    cfg = _cfg()
    f = explicit_fractions(LABEL_NAMES, 20, cfg["groups"], cfg["families"], p)
    assert f.shape == (20, 18) and np.allclose(f.sum(axis=0), 1.0)
    assert f[0, C["DDoS-ICMP_Flood"]] == pytest.approx(ddos_in)
    assert f[5, C["Recon-PortScan"]] == pytest.approx(ddos_in)
    assert f[10, C["XSS"]] == pytest.approx(web_in)
    assert f[14, C["Mirai-udpplain"]] == pytest.approx(other_in)
    assert f[19, C["DDoS-ICMP_Flood"]] == pytest.approx(out) and f[0, C["XSS"]] == pytest.approx(out)
    assert np.allclose(f[:, C["Benign"]], 0.05)


def test_counts_floor_plus_largest_remainder_with_low_index_ties():
    assert explicit_counts(np.array([[0.5], [0.25], [0.25]]), np.array([5])).tolist() == [[3], [1], [1]]
    assert explicit_counts(np.full((3, 1), 1 / 3), np.array([4])).tolist() == [[2], [1], [1]]


def test_partition_is_deterministic_and_matches_the_count_table():
    ids, labels = _data()
    a = partition_explicit(ids, labels, LABEL_NAMES, _cfg(seed=1))
    b = partition_explicit(ids, labels, LABEL_NAMES, _cfg(seed=2))
    assert all(np.array_equal(x, y) for x, y in zip(a.client_flow_ids, b.client_flow_ids))
    cfg = _cfg()
    expected = explicit_counts(explicit_fractions(LABEL_NAMES, 20, cfg["groups"], cfg["families"], 0.8), np.full(18, 400))
    assert np.array_equal(a.counts, expected) and np.array_equal(a.class_histogram(labels, 18), expected)
    flat = np.concatenate(a.client_flow_ids)
    assert flat.size == np.unique(flat).size and set(flat.tolist()) <= set(ids.tolist())


def test_clients_get_contiguous_time_segments():
    ids, labels = _data()
    order = -ids.astype(np.float64)  # later ids are earlier in time
    part = partition_explicit(ids, labels, LABEL_NAMES, _cfg(), order=order)
    for c in range(18):
        runs = [np.sort(k[labels[k] == c])[::-1] for k in part.client_flow_ids]
        assert np.array_equal(np.concatenate(runs), np.sort(ids[labels == c])[::-1])
    assert part.params["ordered_by"] == "flow_start_ts"


def test_zero_dominance_is_an_even_split():
    ids, labels = _data(per_class=203)
    part = partition_explicit(ids, labels, LABEL_NAMES, _cfg(dominance=0.0))
    assert (part.counts[:3] == 11).all() and (part.counts[3:] == 10).all()


def test_cap_keeps_the_earliest_flows_of_each_run():
    ids, labels = _data()
    full = partition_explicit(ids, labels, LABEL_NAMES, _cfg())
    capped = partition_explicit(ids, labels, LABEL_NAMES, _cfg(max_flows_per_client=60))
    for k, (f_ids, c_ids) in enumerate(zip(full.client_flow_ids, capped.client_flow_ids)):
        assert c_ids.size == min(f_ids.size, 60) and capped.counts[k].sum() == c_ids.size
        for c in range(18):
            run, kept = np.sort(f_ids[labels[f_ids] == c]), np.sort(c_ids[labels[c_ids] == c])
            assert np.array_equal(kept, run[: kept.size])


def test_class_outside_every_family_is_split_evenly():
    names = {**LABEL_NAMES, 18: "Weird"}
    cfg = _cfg()
    f = explicit_fractions(names, 20, cfg["groups"], cfg["families"], 0.8)
    assert np.allclose(f[:, 18], 0.05)
    ids, labels = _data(extra=("Weird",))
    part = partition_explicit(ids, labels, names, cfg)
    assert part.counts[:, 18].sum() == 400


@pytest.mark.parametrize(
    "override",
    [
        {"groups": {"A": {"clients": [1, 10], "family": None}, "B": {"clients": [10, 20], "family": None}}},
        {"groups": {"A": {"clients": [1, 19], "family": None}}},
        {"groups": {"A": {"clients": [1, 20], "family": "nope"}}},
        {"dominance": 1.5},
        {"min_flows_per_client": 1000},
    ],
)
def test_invalid_explicit_config_raises(override):
    ids, labels = _data()
    with pytest.raises(ValueError):
        partition_explicit(ids, labels, LABEL_NAMES, _cfg(**override))


def test_make_partition_explicit_needs_label_names_and_json_round_trip():
    ids, labels = _data()
    with pytest.raises(ValueError):
        make_partition(ids, labels, _cfg())
    part = make_partition(ids, labels, _cfg(), label_names=LABEL_NAMES)
    back = ClientPartition.from_json(part.to_json())
    assert np.array_equal(back.counts, part.counts) and back.params["dominance"] == 0.8


def test_select_malicious_group_placement():
    groups = load_fl_config(None)["partition"]["groups"]
    assert select_malicious(20, 0.2, 0, placement="group:G2", groups=groups) == [5, 6, 7, 8]
    with pytest.raises(ValueError):
        select_malicious(20, 0.3, 0, placement="group:G2", groups=groups)
    with pytest.raises(ValueError):
        select_malicious(20, 0.2, 0, placement="group:G9", groups=groups)
    with pytest.raises(ValueError):
        select_malicious(20, 0.2, 0, placement="bogus")
```

Append to `tests/federated/test_simulation.py`:

```python
TINY_GROUPS = {
    "G1": {"clients": [1, 1], "family": "ddos"},
    "G2": {"clients": [2, 2], "family": "web"},
    "G3": {"clients": [3, 3], "family": None},
}


def test_fl_smoke_explicit_partition_and_group_placement(smoke_cfg, tmp_path):
    cfg = smoke_cfg(
        "explicit",
        **{
            "partition.scheme": "explicit",
            "partition.groups": TINY_GROUPS,
            "attack.name": "sign_flip",
            "attack.malicious_ratio": 0.34,
            "attack.placement": "group:G2",
        },
    )
    summary = run_experiment(cfg)
    assert summary["malicious_clients"] == [1]
    part = json.loads((tmp_path / "explicit" / "partition.json").read_text(encoding="utf-8"))
    assert part["scheme"] == "explicit" and len(part["counts"]) == 3 and len(part["counts"][0]) == 3


def test_export_partition_table_script(smoke_cfg, tmp_path):
    import importlib.util
    from pathlib import Path

    import yaml

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(smoke_cfg("x", **{"partition.scheme": "explicit", "partition.groups": TINY_GROUPS})), encoding="utf-8")
    script = Path(__file__).resolve().parents[2] / "scripts" / "fl" / "export_partition_table.py"
    spec = importlib.util.spec_from_file_location("export_partition_table", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out_csv = tmp_path / "table.csv"
    assert module.main(["--config", str(cfg_path), "--out-csv", str(out_csv), "--out-png", str(tmp_path / "t.png")]) == 0
    lines = out_csv.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "client,group,Benign,XSS,DDoS-ICMP_Flood,total" and len(lines) == 4
    pytest.importorskip("matplotlib")
    assert (tmp_path / "t.png").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PY -m pytest tests/federated/test_partition_explicit.py tests/federated/test_simulation.py -q`
Expected: FAIL — `ImportError: cannot import name 'explicit_counts'`.

- [ ] **Step 3: Implement the explicit scheme in `partition.py` and extend `make_partition`**

- [ ] **Step 4: Update `config.py` defaults (block above) and `simulation.py`**

`run_experiment` passes `label_names=ctx.label_names, order=ctx.flow_order` to `make_partition`. `select_malicious`: `random` keeps the revision-1 rule (`m = round(ratio·K)`, `default_rng(seed + 7919)`); `group:<name>` takes the first `m` members of that group in client order (`ValueError` if the group is unknown, `groups` is `None`, or `m` exceeds its size); anything else ⇒ `ValueError`. `run_experiment` validates `attack.placement` before training.

- [ ] **Step 5: Implement `scripts/fl/export_partition_table.py`**

`main(argv) -> int` with `--config`, repeatable `--set KEY=VALUE`, `--out-csv`, `--out-png`. Inserts `src/` on `sys.path`, loads the config + overrides, `load_context`, `make_partition(..., label_names, order)`, writes CSV header `client,group,<label names in class-id order>,total` and one row per client (`client` 1-based, `group` from `group_members`, empty for non-explicit schemes). Heatmap with matplotlib (`imshow` of counts, class names on x, clients on y, annotated counts); if matplotlib is missing, print a note and still return 0.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `PY -m pytest tests/federated -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/graphslm_ids/offline/federated/{partition,config,simulation}.py scripts/fl/export_partition_table.py tests/federated/test_partition_explicit.py tests/federated/test_simulation.py
git commit -m "feat(fl): explicit gateway class-share partition, group attacker placement, partition table export"
```

---

### Task 11: Baseline models `FlowMLP` and `ByteCNN1D`

**Files:**
- Modify: `src/graphslm_ids/offline/federated/models.py`, `config.py`, `tests/federated/conftest.py`
- Test: `tests/federated/test_models.py`, `tests/federated/test_simulation.py` (append)

**Interfaces:**
- Consumes: T8 `ModelAdapter`, `StepOutput`, `build_adapter`, `rebuild_model`, `LocalTrainer`; `ctx.packet_owner`; trainer `_compute_train_loss`.
- Produces:
  - `FlowMLP(in_dim: int, num_classes: int, hidden_dims=(256, 128, 64), dropout=0.2)`: Linear+ReLU+Dropout per hidden size, final layer attribute named `classifier`.
  - `ByteCNN1D(input_len: int, num_classes: int, channels=(32, 64, 128), kernel_size=7, num_groups=8, pool=(3, 2), dropout=0.2)`: three blocks `Conv1d(k, padding=k//2) → GroupNorm(min(num_groups, ch), ch) → ReLU → MaxPool1d(*pool)`, global max pool over length, `Dropout`, `classifier = Linear(channels[-1], num_classes)`. Input `(B, 1, input_len)`; `ValueError` in the constructor when `input_len` is too short to leave length ≥ 1 after the three pools (the tiny fixture uses the default `num_packets = 8` × 6 byte columns = 48).
  - `flow_packet_csr(owner: int64[P], num_flows: int) -> (ptr: int64[F+1], idx: int64[P'])` (packets of a flow in ascending packet id, which equals `next_packet` order because `graph_builder` links rows `i−1 → i` inside a flow; owner `−1` skipped).
  - `byte_columns(metadata: dict, prefix: str = "obyte_") -> np.ndarray` (column indices in `metadata["packet_feature_names"]`; `ValueError` if the key is missing or no column matches).
  - `flow_byte_matrix(flow_ids, ptr, idx, packet_x, byte_cols, num_packets) -> float32[n, num_packets·len(byte_cols)]` (first `num_packets` packets, zero padding).
  - `MLPAdapter`, `CNNAdapter`; `build_adapter` accepts `mlp`, `cnn`; `rebuild_model` handles both from `ckpt["model_name"]`, `ckpt["config"]["model"]`, `ckpt["num_classes"]` and `checkpoint_meta` keys (`input_dim` for MLP, `input_len` for CNN).
  - Config: `model.mlp: {hidden_dims: [256, 128, 64], dropout: 0.2}`, `model.cnn: {num_packets: 8, byte_prefix: "obyte_", channels: [32, 64, 128], kernel_size: 7, num_groups: 8, dropout: 0.2}`.

Adapter behaviour: `client_data` returns `TensorClientData(x: float32 Tensor, y: int64 Tensor)` — MLP: `(flow_x[ids] − mean) / std` with the round-0 `flow_stats`; CNN: `flow_byte_matrix(...)` stored as float16 (cast to float32 per batch). `train_batches`: `DataLoader(TensorDataset(x, y, ids), batch_size=client.batch_seed_flows, shuffle=True, generator=torch.Generator().manual_seed(seed))`. `forward_loss`: logits → `_compute_train_loss(logits, y, class_weights, loss_type=..., label_smoothing=..., focal_gamma=...)` (the same focal/CB settings as HGT; `eacs` is ignored). `predict`: batches of `eval.batch_seed_flows`, returns ids in input order.

- [ ] **Step 1: Give the tiny fixture packet feature names, then write the failing tests**

In `write_tiny_v3`'s metadata dict add `"packet_feature_names": ["pkt_len", "pkt_dir"] + [f"obyte_{i:03d}" for i in range(6)]` (8 names = `packet_x` width).

`tests/federated/test_models.py`:

```python
import numpy as np
import pytest
import torch

from graphslm_ids.offline.federated.context import federated_flow_stats, load_context
from graphslm_ids.offline.federated.local_trainer import ClientState, LocalTrainer, local_class_weights
from graphslm_ids.offline.federated.models import (
    ByteCNN1D,
    FlowMLP,
    build_adapter,
    byte_columns,
    flow_byte_matrix,
    flow_packet_csr,
)
from graphslm_ids.offline.federated.params import ParamLayout


def test_flow_mlp_shape_and_groups():
    model = FlowMLP(6, 3, hidden_dims=(16, 8), dropout=0.0)
    assert model(torch.randn(4, 6)).shape == (4, 3)
    assert set(ParamLayout.from_model(model).group_names) == {"other", "classifier"}


def test_byte_cnn_shape_and_no_batchnorm():
    model = ByteCNN1D(48, 3, channels=(4, 8, 8), kernel_size=7, num_groups=2, dropout=0.0)
    assert model(torch.randn(2, 1, 48)).shape == (2, 3)
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
    assert set(ParamLayout.from_model(model).group_names) == {"other", "classifier"}


def test_flow_packet_csr_and_byte_matrix():
    owner = np.array([0, 0, 2, 2, 2, -1])
    ptr, idx = flow_packet_csr(owner, num_flows=3)
    assert ptr.tolist() == [0, 2, 2, 5] and idx.tolist() == [0, 1, 2, 3, 4]
    packet_x = np.arange(24, dtype=np.float16).reshape(6, 4)
    m = flow_byte_matrix(np.array([2, 1, 0]), ptr, idx, packet_x, np.array([1, 3]), num_packets=2)
    assert m.dtype == np.float32 and m.shape == (3, 4)
    assert m[0].tolist() == [9, 11, 13, 15] and m[1].tolist() == [0, 0, 0, 0] and m[2].tolist() == [1, 3, 5, 7]
    padded = flow_byte_matrix(np.array([0]), ptr, idx, packet_x, np.array([1, 3]), num_packets=3)
    assert padded[0].tolist() == [1, 3, 5, 7, 0, 0]


def test_byte_columns():
    assert byte_columns({"packet_feature_names": ["a", "obyte_000", "obyte_001"]}).tolist() == [1, 2]
    with pytest.raises(ValueError):
        byte_columns({})
    with pytest.raises(ValueError):
        byte_columns({"packet_feature_names": ["a"]})


@pytest.mark.parametrize("name", ["mlp", "cnn"])
def test_baseline_adapters_train_and_predict(smoke_cfg, name):
    cfg = smoke_cfg(name, **{"model.name": name, "model.cnn.num_groups": 2})
    ctx = load_context(cfg)
    ids = np.arange(30, dtype=np.int64)
    adapter = build_adapter(cfg, ctx, federated_flow_stats(ctx.artifact.node_features["flow"], [ids]), torch.device("cpu"))
    torch.manual_seed(0)
    model = adapter.build_model()
    layout = ParamLayout.from_model(model)
    labels = ctx.labels[ids]
    state = ClientState(0, ids, labels, adapter.client_data(0, ids, seed=0), local_class_weights(labels, 3))
    update = LocalTrainer(model, layout, adapter, "cpu", cfg["client"]).train(state, layout.flatten(model.state_dict()), 1, 1e-2, 0)
    assert bool(torch.isfinite(update.delta).all()) and float(update.delta.abs().sum()) > 0
    gids, logits = adapter.predict(model, ctx.test_ids)
    assert gids.tolist() == ctx.test_ids.tolist() and tuple(logits.shape) == (len(ctx.test_ids), 3)
    assert adapter.checkpoint_meta()["model_name"] == name
```

Append to `tests/federated/test_simulation.py`:

```python
@pytest.mark.parametrize("name", ["mlp", "cnn"])
def test_fl_smoke_baseline_models(smoke_cfg, tmp_path, name):
    cfg = smoke_cfg(
        name,
        **{"model.name": name, "model.cnn.num_groups": 2,
           "attack.name": "sign_flip", "attack.malicious_ratio": 0.34},
    )
    summary = run_experiment(cfg)
    assert summary["model"] == name and len(summary["malicious_clients"]) == 1
    ckpt = torch.load(tmp_path / name / "global_final.pt", map_location="cpu", weights_only=False)
    rebuild_model(ckpt).load_state_dict(ckpt["model_state_dict"])
    for mode in ("centralized", "local_only"):
        assert run_experiment(apply_overrides(cfg, {"federated.mode": mode, "output.dir": str(tmp_path / f"{name}_{mode}")}))["mode"] == mode
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PY -m pytest tests/federated/test_models.py -q`
Expected: FAIL — `ImportError: cannot import name 'FlowMLP'`.

- [ ] **Step 3: Implement the networks, helpers and both adapters in `models.py`; add the config defaults**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PY -m pytest tests/federated -q`
Expected: all pass (including the earlier client-graph tests with the new fixture metadata key).

- [ ] **Step 5: Commit**

```bash
git add src/graphslm_ids/offline/federated/{models,config}.py tests/federated/{conftest,test_models,test_simulation}.py
git commit -m "feat(fl): FlowMLP and ByteCNN1D baselines behind the model adapter"
```

---

### Task 12: Comparison aggregators (Median, Trimmed Mean, Multi-Krum, SignGuard, DnC)

**Files:**
- Modify: `src/graphslm_ids/offline/federated/aggregators.py`
- Test: `tests/federated/test_aggregators.py` (append), `tests/federated/test_server.py` (append)

**Interfaces:**
- Consumes: `linalg.{coordinate_median, coordinate_trimmed_mean, pairwise_sq_dists, top_right_singular_vector}`.
- Produces: `Median()`, `TrimmedMean(trim_ratio=0.2, trim_k=None)`, `MultiKrum(f=None, f_ratio=0.2, m=None)`, `SignGuard(lower=0.1, upper=3.0, coord_ratio=0.1, quantile=0.5, seed=0)`, `DnC(niters=1, c=1.0, b=10000, f=None, f_ratio=0.2, seed=0)`; helpers `largest_meanshift_cluster`, `sign_features`. `build_aggregator` defaults: `trimmed_mean.trim_ratio`, `multi_krum.f_ratio`, `dnc.f_ratio` ← `assumed_malicious_ratio`; `signguard.seed`, `dnc.seed` ← `seed`. Registry names: `fedavg, median, trimmed_mean, multi_krum, signguard, dnc`. MSGuard is optional in the spec and stays out of this plan.

Definitions are spec §3.6 as originally published (no clipping, no relation-aware wrapper). Reference sketch: revision-1 Task 7 classes of the same names (drop the `variant="sim"` branch of SignGuard). Filtering rules (Multi-Krum, SignGuard, DnC) return `selected`. `TrimmedMean` raises `ValueError` when `2k ≥ N`; `MultiKrum` when `N ≤ 2f + 2`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/federated/test_aggregators.py` the revision-1 Task 7 tests `test_median_odd_and_even`, `test_trimmed_mean_drops_extremes_and_rejects_over_trim`, `test_multi_krum_requires_enough_clients`, and `test_build_aggregator_defaults_and_errors` (unchanged), `test_robust_rules_resist_a_large_outlier` and `test_filtering_rules_report_the_outlier_as_not_selected` parametrized without `MSGuard`, plus:

```python
def test_filtering_rules_drop_a_sign_flipped_cluster():
    g = torch.Generator().manual_seed(1)
    base = torch.rand(200, generator=g) + 0.5  # all-positive mean: sign statistics separate the flip
    x = base + 0.1 * torch.randn(10, 200, generator=g)
    x[:3] = -x[:3]
    for agg in (MultiKrum(f_ratio=0.3), SignGuard(seed=0), DnC(f_ratio=0.3, b=50, seed=0)):
        sel = agg.aggregate(x, torch.ones(10)).selected
        assert not bool(sel[:3].any()) and bool(sel[3:].all()), agg.name
```

Append to `tests/federated/test_server.py`:

```python
def test_server_marks_multi_krum_dropped_clients_rejected():
    from graphslm_ids.offline.federated.aggregators import MultiKrum

    g = torch.Generator().manual_seed(0)
    deltas = torch.randn(20, generator=g) + 0.1 * torch.randn(10, 20, generator=g)
    deltas[0] = -50.0 * deltas[1]
    server = FLServer(global_vec=torch.zeros(20), aggregator=MultiKrum(f_ratio=0.2))
    assert server.step([ClientMessage(i, 1, deltas[i], 1) for i in range(10)]).decision_map()[0] == "reject"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PY -m pytest tests/federated/test_aggregators.py -q`
Expected: FAIL — `ImportError: cannot import name 'Median'`.

- [ ] **Step 3: Implement the five rules and the builder defaults**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PY -m pytest tests/federated -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/graphslm_ids/offline/federated/aggregators.py tests/federated/test_{aggregators,server}.py
git commit -m "feat(fl): Median, Trimmed Mean, Multi-Krum, SignGuard and DnC aggregators"
```

---

### Task 13: Proposed defense `RAGuard`

**Files:**
- Create: `src/graphslm_ids/offline/federated/defense.py`
- Modify: `src/graphslm_ids/offline/federated/linalg.py`, `simulation.py`
- Test: `tests/federated/test_defense.py`, `tests/federated/test_linalg.py` (append), `tests/federated/test_simulation.py` (append)

**Interfaces:**
- Consumes: T7 `Aggregator`, `AggResult`, decision constants, `build_aggregator`; `linalg.{coordinate_median, cosine_to}`; `ParamLayout.{group_names, group_coords}`.
- Produces:
  - `linalg.weighted_trimmed_mean(x: Tensor[N, D], w: Tensor[N], beta: float, chunk: int = CHUNK) -> Tensor[D]`.
  - `defense.robust_z_sigma(values: np.ndarray, floor: float) -> np.ndarray` (`(v − median) / max(1.4826·MAD, floor)`).
  - `defense.sign_agreement(x: Tensor[n, d], ref: Tensor[d]) -> np.ndarray` (share of coordinates with `sign(ref) ≠ 0` where `sign(x) == sign(ref)`).
  - `defense.clip_to_median(deltas: Tensor[N, D], factor: float) -> Tensor[N, D]` (each row scaled by `min(1, factor·median_norm / norm)`; median = `float(norms.median())`).
  - `defense.client_suspicion(u: np.ndarray[N, G], evaluated: bool[N, G], group_sizes: np.ndarray[G], reduce: str) -> np.ndarray[N]` (`mean`: parameter-count-weighted mean over evaluated groups; `max`; no evaluated group ⇒ 0).
  - `defense.decide(trust: np.ndarray, *, downweight: bool = True, reject_below: float = 0.5, accept_above: float = 0.8) -> tuple[list[str], np.ndarray]` (weights: 0 / trust / 1; with `downweight=False` everything ≥ `reject_below` is `accept` with weight 1).
  - `defense.RAGuard(Aggregator)`, `name = "raguard"`, constructor kwargs (defaults): `beta=0.2, clip_factor=1.0, z_max=3.0, min_active=3, scale_floor=1e-3, ema_current=0.8, warmup_rounds=2, abrupt_k=3.0, abrupt_min_jump=0.15, reject_below=0.5, accept_above=0.8, relation_aware=True, temporal=True, downweight=True, aggregator="trimmed_mean", client_reduce="mean"`; methods `update_trust(client_ids: list[int], suspicion: np.ndarray, round_idx: int) -> tuple[np.ndarray, np.ndarray]` (trust, abrupt flags) and `aggregate(deltas, weights, ctx) -> AggResult` (with `decisions`, `trust`, `weights`, `selected`, `scores = {"suspicion", "abrupt"}`, `info = {"num_groups_evaluated", ...}`).
  - `defense.build_defense(cfg: dict, *, assumed_malicious_ratio: float = 0.2, seed: int = 0) -> Aggregator` (`raguard` ⇒ `RAGuard(beta=assumed_malicious_ratio, **params)`; anything else ⇒ `build_aggregator`; bad params ⇒ `ValueError`).

`RAGuard.aggregate` (spec §3.7; the parts tests do not pin down):

```
ids, r, layout, prev = ctx["client_ids"], ctx["round_idx"], ctx.get("layout"), ctx.get("prev_aggregate")
groups = [(g, layout.group_coords([g])) for g in layout.group_names] if relation_aware and layout else [("all", arange(D))]
u, evaluated = zeros(N, G), zeros(N, G, bool)
for gi, (g, cols):                                   # signals on RAW deltas, so a x50 update is visible
    sub = deltas[:, cols]; norms = sub.norm(dim=1) (float64); act = norms > 0
    if act.sum() < min_active: continue
    ref = prev[cols] if prev is not None and prev[cols].norm() > 0 else coordinate_median(sub[act])
    z_mag = robust_z_sigma(log(norms[act] / median(norms[act])))
    z_cos = robust_z_sigma(cosine_to(sub[act], ref));  z_sgn = robust_z_sigma(sign_agreement(sub[act], ref))
    s = 1 − (1 − hi(|z_mag|)) (1 − hi(−z_cos)) (1 − hi(−z_sgn)),   hi(z) = clip((z − 1)/(z_max − 1), 0, 1)
    u[act, gi] = s; evaluated[act, gi] = True
U = client_suspicion(u, evaluated, sizes, client_reduce)
trust, abrupt = update_trust(ids, U, r)
decisions, dw = decide(trust, downweight=downweight, ...)
w = weights · dw;  clipped = clip_to_median(deltas, clip_factor);  out = zeros(D)
for gi, (g, cols):                                   # aggregate CLIPPED deltas
    rows = (decision ≠ reject) & (raw group norm > 0)
    if rows.any(): out[cols] = weighted_trimmed_mean(clipped[rows][:, cols], w[rows], beta)   # or coordinate_median
```

`update_trust` keeps per-client `trust_prev`, `u_prev`, and a history of `|U_t − U_{t−1}|`. Rounds `≤ warmup_rounds` or `temporal=False`: `trust = 1 − U`, no abrupt flag. Later rounds: `trust = ema_current·(1 − U) + (1 − ema_current)·trust_prev` (first sighting: `1 − U`); if the client has ≥ 2 history values and `|U − u_prev| > max(median(hist) + abrupt_k·MAD(hist), abrupt_min_jump)`, set `abrupt` and `trust = min(trust, 0.5)`. Always append the new jump to the history afterwards. `abrupt_min_jump` and the 1.4826 MAD factor are our choices (the spec states the rule in σ units without a floor); T16 records them in the spec.

`weighted_trimmed_mean`: per coordinate sort values, carry weights normalized to 1, cumulative mass `c_i`; retained mass of element `i` = `max(0, min(c_i, 1−β) − max(c_{i−1}, β))`; result `= Σ retained·v / (1 − 2β)`; `ValueError` unless `0 ≤ β < 0.5`; chunk over coordinates like `coordinate_trimmed_mean`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/federated/test_linalg.py`:

```python
def test_weighted_trimmed_mean_cases():
    from graphslm_ids.offline.federated.linalg import weighted_trimmed_mean

    x = torch.tensor([[0.0], [1.0], [2.0], [100.0]])
    assert torch.allclose(weighted_trimmed_mean(x, torch.ones(4), 0.25), torch.tensor([1.5]))
    assert torch.allclose(weighted_trimmed_mean(x, torch.tensor([1.0, 1.0, 1.0, 1.0]), 0.0), x.mean(0))
    two = torch.tensor([[0.0], [10.0]])
    assert torch.allclose(weighted_trimmed_mean(two, torch.tensor([3.0, 1.0]), 0.2), torch.tensor([0.5 / 0.6]))
    with pytest.raises(ValueError):
        weighted_trimmed_mean(x, torch.ones(4), 0.5)
```

`tests/federated/test_defense.py`:

```python
import numpy as np
import pytest
import torch

from graphslm_ids.offline.federated.aggregators import Median
from graphslm_ids.offline.federated.defense import RAGuard, build_defense, client_suspicion, clip_to_median, decide
from graphslm_ids.offline.federated.linalg import coordinate_median, weighted_trimmed_mean
from graphslm_ids.offline.federated.params import ParamLayout

LAYOUT = ParamLayout.from_state_dict(
    {
        "layers.0.relation_key.flow__r1__packet.weight": torch.zeros(40),
        "layers.0.relation_key.packet__r2__technique.weight": torch.zeros(40),
        "classifier.weight": torch.zeros(20),
    }
)
R2 = LAYOUT.group_coords(["rel:packet__r2__technique"])


def _round(seed=0, n=20, noise=0.1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(100, generator=g) + noise * torch.randn(n, 100, generator=g)


def _run(guard, x, round_idx=1, prev=None, layout=LAYOUT):
    ctx = {"client_ids": list(range(x.shape[0])), "round_idx": round_idx, "layout": layout, "prev_aggregate": prev}
    return guard.aggregate(x, torch.ones(x.shape[0]), ctx)


def test_decide_thresholds_and_downweight_flag():
    decisions, w = decide(np.array([0.3, 0.6, 0.9]))
    assert decisions == ["reject", "downweight", "accept"] and np.allclose(w, [0.0, 0.6, 1.0])
    decisions, w = decide(np.array([0.3, 0.6, 0.9]), downweight=False)
    assert decisions == ["reject", "accept", "accept"] and np.allclose(w, [0.0, 1.0, 1.0])


def test_client_suspicion_mean_and_max():
    u = np.array([[0.0, 1.0], [0.5, 0.5], [0.9, 0.9]])
    ev = np.array([[True, True], [True, False], [False, False]])
    sizes = np.array([3.0, 1.0])
    assert np.allclose(client_suspicion(u, ev, sizes, "mean"), [0.25, 0.5, 0.0])
    assert np.allclose(client_suspicion(u, ev, sizes, "max"), [1.0, 0.5, 0.0])


def test_rejects_a_scaled_outlier_without_rejecting_benign():
    x = _round()
    x[0] = 50.0 * x[0]
    res = _run(RAGuard(), x)
    assert res.decisions[0] == "reject" and "reject" not in res.decisions[1:]
    assert not bool(res.selected[0]) and res.trust.shape == (20,)


def test_rejects_a_sign_flip():
    x = _round()
    x[0] = -x[0]
    assert _run(RAGuard(), x).decisions[0] == "reject"


def test_zero_relation_group_is_not_penalized_unless_relation_awareness_is_off():
    x = _round()
    x[:6, R2] = 0.0  # non-IID: six benign clients never see relation r2
    assert "reject" not in _run(RAGuard(), x).decisions
    assert "reject" in _run(RAGuard(relation_aware=False), x).decisions[:6]


def test_warmup_uses_cross_sectional_trust_only():
    guard = RAGuard()
    t1, _ = guard.update_trust([0], np.array([0.4]), 1)
    t2, _ = guard.update_trust([0], np.array([0.0]), 2)
    t3, _ = guard.update_trust([0], np.array([0.0]), 3)
    assert t1[0] == pytest.approx(0.6) and t2[0] == pytest.approx(1.0) and t3[0] == pytest.approx(1.0)


def test_abrupt_change_caps_trust_after_warmup():
    guard = RAGuard()
    for r, u in enumerate([0.05, 0.06, 0.05, 0.06], start=1):
        guard.update_trust([0], np.array([u]), r)
    trust, abrupt = guard.update_trust([0], np.array([0.30]), 5)
    assert bool(abrupt[0]) and trust[0] == pytest.approx(0.5)
    plain = RAGuard(temporal=False)
    for r, u in enumerate([0.05, 0.06, 0.05, 0.06, 0.30], start=1):
        trust, abrupt = plain.update_trust([0], np.array([u]), r)
    assert not bool(abrupt[0]) and trust[0] == pytest.approx(0.70)


def test_aggregate_is_weighted_trimmed_mean_or_median_of_clipped_rows():
    x = _round()  # every client is active in every group, so per-group = whole-vector aggregation
    clipped = clip_to_median(x, 1.0)
    res = _run(RAGuard(), x)
    keep = torch.tensor([d != "reject" for d in res.decisions])
    assert bool(keep.all())
    expected = weighted_trimmed_mean(clipped[keep], torch.as_tensor(res.weights, dtype=torch.float32)[keep], 0.2)
    assert torch.allclose(res.delta, expected, atol=1e-6)
    med = _run(RAGuard(aggregator="median"), x)
    assert torch.allclose(med.delta, coordinate_median(clipped), atol=1e-6)


def test_group_nobody_trains_aggregates_to_zero_and_layout_none_is_one_group():
    x = _round()
    x[:, R2] = 0.0
    assert int(torch.count_nonzero(_run(RAGuard(), x).delta[R2])) == 0
    assert _run(RAGuard(), x, layout=None).delta.shape == (100,)


def test_build_defense():
    guard = build_defense({"name": "raguard", "params": {"temporal": False}}, assumed_malicious_ratio=0.3)
    assert isinstance(guard, RAGuard) and guard.beta == 0.3 and guard.temporal is False
    assert isinstance(build_defense({"name": "median"}), Median)
    with pytest.raises(ValueError):
        build_defense({"name": "raguard", "params": {"bogus": 1}})
    with pytest.raises(ValueError):
        RAGuard(beta=0.5)
    with pytest.raises(ValueError):
        RAGuard(aggregator="mean")
```

Append to `tests/federated/test_simulation.py`:

```python
def test_fl_smoke_raguard(smoke_cfg, tmp_path):
    summary = run_experiment(
        smoke_cfg("rag", **{"attack.name": "min_max", "attack.malicious_ratio": 0.34, "defense.aggregator": {"name": "raguard", "params": {}}})
    )
    assert summary["aggregator"] == "raguard"
    first = _records(tmp_path / "rag" / "rounds.jsonl")[0]
    assert all("trust" in d for d in first["decisions"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PY -m pytest tests/federated/test_defense.py tests/federated/test_linalg.py -q`
Expected: FAIL — `ModuleNotFoundError: ... federated.defense`.

- [ ] **Step 3: Implement `weighted_trimmed_mean` in `linalg.py`**

- [ ] **Step 4: Implement `defense.py`**

- [ ] **Step 5: Switch `simulation.py` from `build_aggregator` to `build_defense`**

- [ ] **Step 6: Run the tests to verify they pass**

Run: `PY -m pytest tests/federated -q`
Expected: all pass. If `test_rejects_a_scaled_outlier_without_rejecting_benign` or `test_zero_relation_group_is_not_penalized_unless_relation_awareness_is_off` rejects a benign client, report the trust values instead of changing the test seed: that is hypothesis GT1 failing and the constants need a decision.

- [ ] **Step 7: Commit**

```bash
git add src/graphslm_ids/offline/federated/{defense,linalg,simulation}.py tests/federated/test_{defense,linalg,simulation}.py
git commit -m "feat(fl): RAGuard relation-aware, history-aware screening with per-group trimmed mean"
```

---

### Task 14: Saved deltas and offline detection replay

**Files:**
- Create: `src/graphslm_ids/offline/federated/offline_eval.py`
- Modify: `src/graphslm_ids/offline/federated/params.py`, `simulation.py`, `config.py`
- Test: `tests/federated/test_offline_eval.py`, `tests/federated/test_params.py` (append)

**Interfaces:**
- Consumes: T7 `FLServer`, `ClientMessage`, metrics counts; T13 `build_defense`; `FLServer.last_inputs`.
- Produces:
  - `ParamLayout.from_groups(groups: dict[str, list[tuple[int, int]]], numel: int) -> ParamLayout` (empty `names`/`shapes`/`dtypes`/`offsets`; group methods work) and `ParamLayout.groups_json() -> dict[str, list[list[int]]]`.
  - `offline_eval.save_round_deltas(out_dir: Path, round_idx: int, client_ids: list[int], deltas: Tensor[N, D], num_samples: Tensor[N], malicious: list[int], dtype: str = "float16") -> str` (writes `out_dir/round_{r:03d}.pt` = `{"round", "client_ids", "deltas", "num_samples", "malicious"}`; returns the dtype actually used: `float32` when `float16` would overflow, i.e. `max |x| > 65504`).
  - `offline_eval.load_round_deltas(path) -> dict` (deltas as float32).
  - `offline_eval.write_delta_meta(out_dir, layout, *, weighting, server_lr, malicious_clients, rounds) -> None` (`meta.json` with `numel`, `groups`, and those fields).
  - `offline_eval.replay(run_dir, defense_cfg: dict, *, assumed_malicious_ratio=0.2, seed=42) -> dict`: `{"defense": name, "rounds": [{"round", "decisions": {cid: decision}, "detection": counts}], "detection": {**totals, **rates}}`, built by stepping a fresh `FLServer(global_vec=zeros(numel), aggregator=build_defense(...), layout=from_groups(...), weighting=meta["weighting"])` through every saved round in order (empty rounds still step).
  - `offline_eval.main(argv) -> int`: `--run DIR`, repeatable `--defense NAME`, optional `--params-json`, `--out PATH` (default `<run>/offline_eval.json`), writes `{name: replay(...)}`.
  - Config: `output.save_deltas: false`, `output.save_deltas_dtype: "float16"`. Simulation (`fl` mode only) writes `deltas/meta.json` once and `deltas/round_XXX.pt` after every `server.step` from `server.last_inputs` (an empty `(0, D)` tensor when no message was valid); the summary records the dtype per round under `cost.saved_delta_dtypes`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/federated/test_params.py`:

```python
def test_layout_from_groups_round_trip():
    layout = ParamLayout.from_state_dict(
        {"layers.0.relation_key.a__r__b.weight": torch.zeros(3), "classifier.weight": torch.zeros(2)}
    )
    back = ParamLayout.from_groups({g: [tuple(s) for s in spans] for g, spans in layout.groups_json().items()}, layout.numel)
    assert back.group_names == layout.group_names and back.numel == 5
    assert torch.equal(back.group_coords(["classifier"]), layout.group_coords(["classifier"]))
```

`tests/federated/test_offline_eval.py`:

```python
import json

import pytest
import torch

from graphslm_ids.offline.federated.offline_eval import load_round_deltas, main, replay, save_round_deltas
from graphslm_ids.offline.federated.simulation import run_experiment


def test_float16_overflow_falls_back_to_float32(tmp_path):
    used = save_round_deltas(tmp_path, 1, [0, 1], torch.tensor([[1e6, 0.5], [0.1, 0.2]]), torch.tensor([1.0, 1.0]), [0], "float16")
    assert used == "float32"
    rec = load_round_deltas(tmp_path / "round_001.pt")
    assert bool(torch.isfinite(rec["deltas"]).all()) and float(rec["deltas"][0, 0]) == 1e6
    assert save_round_deltas(tmp_path, 2, [0], torch.tensor([[0.5, 0.25]]), torch.tensor([1.0]), [], "float16") == "float16"


def _live(smoke_cfg, tmp_path):
    cfg = smoke_cfg(
        "live",
        **{
            "attack.name": "sign_flip",
            "attack.malicious_ratio": 0.34,
            "defense.aggregator": {"name": "raguard", "params": {}},
            "output.save_deltas": True,
            "output.save_deltas_dtype": "float32",
            "federated.rounds": 3,
        },
    )
    run_experiment(cfg)
    return tmp_path / "live"


def test_replay_reproduces_live_decisions(smoke_cfg, tmp_path):
    run = _live(smoke_cfg, tmp_path)
    live = [json.loads(line) for line in (run / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]
    out = replay(run, {"name": "raguard", "params": {}})
    assert [r["round"] for r in out["rounds"]] == [1, 2, 3]
    for rec, rep in zip(live, out["rounds"]):
        assert {int(d["client_id"]): d["decision"] for d in rec["decisions"]} == {int(k): v for k, v in rep["decisions"].items()}
    assert out["detection"]["tp"] + out["detection"]["fn"] == 3


def test_offline_eval_cli_scores_several_defenses(smoke_cfg, tmp_path):
    run = _live(smoke_cfg, tmp_path)
    assert main(["--run", str(run), "--defense", "median", "--defense", "raguard"]) == 0
    result = json.loads((run / "offline_eval.json").read_text(encoding="utf-8"))
    assert set(result) == {"median", "raguard"} and result["median"]["detection"]["tpr"] is not None
    with pytest.raises(ValueError):
        replay(run, {"name": "bogus"})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PY -m pytest tests/federated/test_offline_eval.py tests/federated/test_params.py -q`
Expected: FAIL — `ModuleNotFoundError: ... federated.offline_eval`.

- [ ] **Step 3: Implement `ParamLayout.from_groups` / `groups_json`, `offline_eval.py`, the config keys and the simulation hook**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PY -m pytest tests/federated -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/graphslm_ids/offline/federated/{offline_eval,params,simulation,config}.py tests/federated/test_{offline_eval,params}.py
git commit -m "feat(fl): save per-round deltas and replay detection offline for any defense"
```

---

### Task 15: Experiment matrix, scripts and `configs/fl/`

**Files:**
- Create: `src/graphslm_ids/offline/federated/matrix.py`, `scripts/fl/run_matrix.py`, `scripts/fl/summarize_matrix.py`
- Create: `configs/fl/fl_base.yaml` and the grids in the table below
- Test: `tests/federated/test_matrix.py`, `tests/federated/test_configs.py`

**Interfaces:**
- Consumes: T9 `run_experiment`, T8 `load_context`/`data_key`, `config.yaml_load`, T13 `build_defense`, `attacks.build_attack`, T10 `select_malicious`, `group_members`.
- Produces: `expand_grid(grid) -> list[{"name", "labels", "overrides"}]`, `run_grid(grid_path, *, device=None, dry_run=False, only=None, fail_fast=False) -> list[{"name", "status"}]` (`planned` / `skipped` / `done` / `failed`), `summary_row(summary) -> dict`, `add_attack_impact(rows) -> rows`, `summarize(root) -> list[dict]`, `write_csv(rows, path)`, `pivot_markdown(rows, value=..., row_keys=..., col_key=..., group_key=...) -> str`; scripts `main(argv) -> int`.

Reference sketch: revision-1 Task 14, changed as follows:
- `expand_grid` cell order: stable sort by `(labels.get("split"), labels.get("model"), labels.get("partition"), labels.get("seed"))` so one context and one client cache serve consecutive cells.
- `run_grid`: a cell with `fl_summary.json` is `skipped`; otherwise it runs, and on success any stale `error.txt` in its directory is deleted.
- `summary_row`: columns `model`, `aggregator` (no `screen`), every `label_*`, `final_*`/`best_*` test metrics (noisy and clean), `final_test_benign_far`, `final_test_attack_detection`, family recalls (`final_test_recall_<family>` for each family in the summary), `detect_tpr`, `benign_false_rejection` (= detection `fpr`), `downweight_rate_benign`, `malicious_acceptance`, `convergence_round`, cost columns; `local_only` rows also carry `std`/`min`/`max` of test macro-F1.
- `add_attack_impact`: the reference row of a cell is the `mode == "fl"`, `attack == "none"` row with the same labels apart from `attack`, `malicious_ratio` and `placement`; `impact_<metric> = reference − cell` for `final_test_macro_f1` and `final_test_macro_f1_clean`.
- `summarize_matrix.py` default pivots: `final_test_macro_f1`, `final_test_macro_f1_clean`, `benign_false_rejection`, `detect_tpr`, `impact_final_test_macro_f1`.

`configs/fl/fl_base.yaml`: full config with every default section written out: HGT configuration A (`hidden_dim 64`, `num_layers 2`, `num_heads 4`, `sampler.hops 2`, the revision-1 screen-tier fanouts), `model.mlp`/`model.cnn` defaults, the explicit partition block of T10 with `max_flows_per_client: null`, `federated.rounds 20`, `client.local_epochs 1`, `client.gcl.enabled true`, `client.eacs.enabled false`, `data.clean_eval_labels: outputs/v3_ob/clean_eval_labels.npy`, `eval.max_val_flows 5000`, `eval.monitor val_macro_f1_clean`, `output.save_checkpoints true`, `output.save_deltas false`, `device: cuda`.

Grids (all `base_config: configs/fl/fl_base.yaml`, `output_root: outputs/fl/<grid stem>`; `p0`/`p50`/`p80` set `partition.dominance` to 0 / 0.5 / 0.8; attacks are `none, sign_flip, alie, min_max, min_sum, scale_sign`; "compact" means `overrides: {partition.max_flows_per_client: 2000}` — the value is a placeholder to fix after the pilot):

| File | Priority | Axes | Cells |
|---|---|---|---|
| `matrix_pilot.yaml` | pilot | `hgt_config` {A, B (= `model.hidden_dim 128, num_layers 4, num_heads 8, sampler.hops 4` + revision-1 full-tier fanouts)} × `local_epochs` {1, 2} × `partition` {p0, p80}; full data | 8 |
| `matrix_ch1.yaml` | P0 | `mode` {centralized, local_only, fl} × `model` {hgt, mlp, cnn} × `partition` {p0, p50, p80}; full data | 27 |
| `matrix_ch1_temporal.yaml` | P0 | `data.split_protocol: temporal`; HGT; `mode` × `partition` as above | 9 |
| `matrix_p0_defense.yaml` | P0 | HGT; `partition` {p0, p80} × `attack` (6) × `defense` {fedavg, median, trimmed_mean, multi_krum, signguard, dnc, raguard}; `attack.malicious_ratio 0.2`; compact | 84 |
| `matrix_p0_ablation.yaml` | P0 | HGT; p80; `variant` {no_relation, no_temporal, no_downweight, median_agg} (RAGuard params `relation_aware: false` / `temporal: false` / `downweight: false` / `aggregator: median`) × `attack` {none, alie, min_max}; compact | 12 |
| `matrix_p1_ratio.yaml` | P1 | HGT; p80; `defense` {multi_krum, signguard, raguard} × `attack` {alie, min_max} × `malicious_ratio` {0.1, 0.3, 0.4}; compact | 18 |
| `matrix_p1_group.yaml` | P1 | HGT; p80; `attack.placement: group:G2`, ratio 0.2; `defense` (3 as above) × `attack` {alie, min_max}; compact | 6 |
| `matrix_p1_ch2.yaml` | P1 | `model` {mlp, cnn} × `partition` {p0, p80} × `attack` (6) × `defense` {fedavg, multi_krum, signguard, raguard}; compact | 96 |
| `matrix_p1_seeds.yaml` | P1 | HGT; p80; `attack` {none, alie, min_max} × `defense` {fedavg, multi_krum, raguard} × `seed` {43, 44} (`federated.seed` and `partition.seed`); full data | 18 |
| `matrix_p2_eacs.yaml` | P2 | HGT; p80; `client.eacs.enabled true` (+ `anchor_mask_npy: outputs/v3_ob/eacs_anchor_mask.npy`); attack none × `defense` {fedavg, multi_krum, signguard, raguard}; compact | 4 |
| `matrix_p2_threat.yaml` | P2 | HGT; p80; `scenario` {late_start (`attack.start_round 10`), partial (`attack.knowledge partial`)} × `attack` {alie, min_max} × `defense` {multi_krum, raguard}; compact | 8 |

Comment at the top of each P1/P2 grid: the comparison defenses (`multi_krum`, `signguard`) are placeholders for "the two strongest comparison defenses" and must be replaced after the P0 results (spec §3.12).

- [ ] **Step 1: Write the failing tests**

`tests/federated/test_matrix.py`: the revision-1 Task 14 tests (`_grid` helper with the smoke config, `test_expand_grid_dedupes_no_attack_cells`, `test_run_grid_runs_resumes_and_summarizes`, `test_matrix_scripts`), with `"defense": {"fedavg": {"defense.aggregator": {"name": "fedavg"}}}` unchanged, plus:

```python
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
```

`tests/federated/test_configs.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PY -m pytest tests/federated/test_matrix.py tests/federated/test_configs.py -q`
Expected: FAIL — `ModuleNotFoundError: ... federated.matrix`.

- [ ] **Step 3: Implement `matrix.py`, both scripts, `fl_base.yaml` and the eleven grids**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PY -m pytest tests/federated -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/graphslm_ids/offline/federated/matrix.py scripts/fl/run_matrix.py scripts/fl/summarize_matrix.py configs/fl tests/federated/test_{matrix,configs}.py
git commit -m "feat(fl): resumable experiment matrix, summarizer and P0/P1/P2 grids"
```

---

### Task 16: Documentation and full verification

**Files:**
- Modify: `CLAUDE.md`, `docs/superpowers/specs/2026-10-02-fl-hgt-poisoning-testbed-design.md`

- [ ] **Step 1: Add the FL section to `CLAUDE.md`**

Insert before `## Diagnosis history (v1, May 2026)` a section "Federated-learning poisoning testbed (FL-HGT)" with: links to the spec and this plan; one paragraph on what the package does (20 clients on the explicit gateway table, HGT/MLP/CNN behind `ModelAdapter`, five attacks, six comparison defenses + `RAGuard`, `fl`/`centralized`/`local_only`, offline replay; `train_hgt_flow_classifier.py` untouched); the commands below; the per-run output files; and the venv note (`D:\v\nt114`, Python 3.14, torch 2.14 CPU; `tests/baselines` needs `torch_geometric`).

```bat
:: Partition table for the thesis (CSV + heatmap)
D:\v\nt114\Scripts\python.exe scripts/fl/export_partition_table.py --config configs/fl/fl_base.yaml ^
  --set partition.dominance=0.8 --out-csv outputs/fl/partition_p80.csv --out-png outputs/fl/partition_p80.png
:: One experiment
D:\v\nt114\Scripts\python.exe -m graphslm_ids.offline.federated.cli --config configs/fl/fl_base.yaml --device cuda ^
  --set attack.name=alie --set attack.malicious_ratio=0.2 --set defense.aggregator.name=raguard
:: Pilot, then P0 grids (resumable), then tables
D:\v\nt114\Scripts\python.exe scripts/fl/run_matrix.py --grid configs/fl/matrix_pilot.yaml --device cuda
D:\v\nt114\Scripts\python.exe scripts/fl/run_matrix.py --grid configs/fl/matrix_p0_defense.yaml --device cuda
D:\v\nt114\Scripts\python.exe scripts/fl/summarize_matrix.py --root outputs/fl/matrix_p0_defense
:: Offline detection replay on a FedAvg run saved with output.save_deltas=true
D:\v\nt114\Scripts\python.exe -m graphslm_ids.offline.federated.offline_eval --run outputs/fl/<run> ^
  --defense median --defense trimmed_mean --defense multi_krum --defense signguard --defense dnc --defense raguard
:: FL tests (CPU)
D:\v\nt114\Scripts\python.exe -m pytest tests/federated -q
```

- [ ] **Step 2: Update the spec**

§3 file tree: add `context.py` and `hgt_adapter.py` and mark every implemented file `[đã code]`; §3.7: record the σ-scaled MAD (1.4826), signals on raw deltas with aggregation on clipped deltas, and `abrupt_min_jump = 0.15`; §7: mark T7–T16 done with the final `tests/federated` count.

- [ ] **Step 3: Run the FL suite and the whole repository suite**

Run: `PY -m pytest tests/federated -q` then `PY -m pytest tests -q --ignore=tests/baselines`
Expected: all pass; the repository total is 478 + the new FL tests, still 3 skipped.

- [ ] **Step 4: Smoke the CLI for all three models outside pytest**

```bash
PYTHONPATH=src PY - <<'EOF'
import sys, tempfile, pathlib
sys.path.insert(0, "tests")
from federated.conftest import write_tiny_v3
from graphslm_ids.offline.federated.config import apply_overrides, load_fl_config
from graphslm_ids.offline.federated.simulation import run_experiment
root = pathlib.Path(tempfile.mkdtemp())
p = write_tiny_v3(root / "tiny")
groups = {"G1": {"clients": [1, 1], "family": "ddos"}, "G2": {"clients": [2, 2], "family": "web"}, "G3": {"clients": [3, 3], "family": None}}
for name in ("hgt", "mlp", "cnn"):
    cfg = apply_overrides(load_fl_config("configs/fl/fl_base.yaml"), {
        "data.graph_npz": str(p["graph_npz"]), "data.graph_meta_json": str(p["graph_meta_json"]),
        "data.splits_json": str(p["splits_json"]), "data.clean_eval_labels": str(p["clean_eval_labels"]),
        "client.gcl.class_technique_map_csv": str(p["class_technique_map_csv"]),
        "model.name": name, "model.cnn.num_groups": 2,
        "partition.num_clients": 3, "partition.groups": groups, "partition.min_flows_per_client": 1,
        "federated.rounds": 2, "attack.name": "scale_sign", "attack.malicious_ratio": 0.34,
        "defense.aggregator": {"name": "raguard", "params": {}}, "eval.max_val_flows": None,
        "output.dir": str(root / name), "device": "cpu"})
    s = run_experiment(cfg)
    print("OK", name, s["attack"], s["aggregator"], s["final"]["test"]["macro_f1"])
EOF
```

Expected: three lines `OK <model> scale_sign raguard <float>`.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git add -f docs/superpowers/specs/2026-10-02-fl-hgt-poisoning-testbed-design.md docs/superpowers/plans/2026-10-02-fl-hgt-poisoning-testbed.md
git commit -m "docs(fl): document the FL-HGT testbed, RAGuard constants and run commands"
```

---

## Self-review notes

- Spec coverage: §0 requirements 1 (baseline: `centralized`/`local_only`/FedAvg) → T9, T15 `matrix_ch1`; 2 (defense) → T13; 3 (explicit table) → T10; 4 (20 × 20) → T7 defaults, T15 base; 5 (MLP, CNN) → T11; 6 (FL vs no FL) → T9, `matrix_ch1`. §3.1 → T10; §3.2 → T8 (HGT), T11 (MLP/CNN); §3.3 → existing + T11 group check + T14 `from_groups`; §3.4 → T8, T11; §3.5 → existing; §3.6 → T7, T12; §3.7 → T13; §3.8 → T7 (+ T14 delta saving); §3.9 → T9, T10 (placement); §3.10 → T7 metrics, T14 offline, T15 impact; §3.11 → T9, T14; §3.12 → T15; §4 errors → validation tests in T7–T15; §5 tests → per task; §6 limitations → T16 docs; §7 order → task numbering; §8 open items → placeholders flagged in T15 configs.
- Deliberate gaps: MSGuard (optional in the spec, pending the lecturer) and the PMI ablation (needs a separately built artifact) are not planned.
- Every test above names only functions that some task's Interfaces block defines.
