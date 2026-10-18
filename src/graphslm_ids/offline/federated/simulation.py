"""FL experiment driver: ``fl`` / ``centralized`` / ``local_only`` on one v3 artifact.

Round protocol: select participants -> local training on each client's own graph ->
experiment-only Byzantine hook replaces attacker deltas -> the server aggregates and
decides per client -> decisions are graded against ground truth -> periodic validation of
the global model on the full graph (experiment-only).

Everything model-specific lives behind :class:`~graphslm_ids.offline.federated.models.ModelAdapter`;
this module never touches a network class. Configuration is validated up front (mode,
monitor, unknown keys, model / attack / defense names, ...), so a typo fails before the run
trains or writes anything. The headline result is the final-round global model; the
best-validation round is reported alongside.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from graphslm_ids.offline.federated.attacks import Attack, AttackContext, build_attack
from graphslm_ids.offline.federated.config import apply_overrides, unknown_keys
from graphslm_ids.offline.federated.context import (
    ExperimentContext,
    data_key,
    federated_flow_stats,
    load_context,
)
from graphslm_ids.offline.federated.defense import build_defense
from graphslm_ids.offline.federated.local_trainer import (
    ClientState,
    LocalTrainer,
    LocalUpdate,
    local_class_weights,
    round_lr,
)
from graphslm_ids.offline.federated.metrics import (
    compact_metrics,
    convergence_round,
    detection_counts,
    detection_rates,
    evaluate_predictions,
    merge_counts,
)
from graphslm_ids.offline.federated.models import ModelAdapter, build_adapter
from graphslm_ids.offline.federated.offline_eval import save_round_deltas, write_delta_meta
from graphslm_ids.offline.federated.params import ParamLayout
from graphslm_ids.offline.federated.partition import (
    ClientPartition,
    check_partition_config,
    group_members,
    make_partition,
    subsample_partition,
)
from graphslm_ids.offline.federated.records import append_jsonl, to_jsonable, write_json
from graphslm_ids.offline.federated.server import ClientMessage, FLServer
from graphslm_ids.offline.training.train_hgt_flow_classifier import resolve_device, set_seed

VALID_MODES = ("fl", "centralized", "local_only")
VALID_MONITORS = ("val_macro_f1", "val_macro_f1_clean", "val_loss")

# The centralized upper bound: one silo, no adversary, plain FedAvg, always participating,
# exactly one full epoch per round and an unscaled server step.
CENTRALIZED_OVERRIDES: dict[str, Any] = {
    "client.optimizer_state": "persist",
    "client.local_epochs": 1,
    "client.max_local_steps": None,
    "attack.name": "none",
    "attack.malicious_ratio": 0.0,
    "defense.aggregator": {"name": "fedavg", "params": {}},
    "federated.participation": 1.0,
    "federated.server_lr": 1.0,
}
# Local-only clients train alone for every round, so Adam moments carry across rounds.
LOCAL_ONLY_OVERRIDES: dict[str, Any] = {"client.optimizer_state": "persist"}


# -- selection --------------------------------------------------------------------------------


def select_participants(num_clients: int, fraction: float, seed: int, round_idx: int) -> list[int]:
    k = max(1, int(round(float(fraction) * num_clients)))
    if k >= num_clients:
        return list(range(num_clients))
    rng = np.random.default_rng(int(seed) * 100_003 + int(round_idx))
    return sorted(int(c) for c in rng.choice(num_clients, size=k, replace=False))


def parse_placement(placement: Any) -> tuple[str, str | None]:
    """``attack.placement`` as ``("random", None)`` or ``("group", <group name>)``."""
    text = str(placement)
    if text == "random":
        return "random", None
    if text.startswith("group:") and len(text) > len("group:"):
        return "group", text[len("group:") :]
    raise ValueError(f"unknown attack.placement {placement!r}; expected 'random' or 'group:<name>'")


def select_malicious(
    num_clients: int,
    ratio: float,
    seed: int,
    *,
    placement: str = "random",
    groups: dict[str, Any] | None = None,
) -> list[int]:
    """The ``round(ratio * K)`` attacker client ids.

    ``random`` draws them from all clients with ``seed``; ``group:<name>`` takes the first ones,
    in client order, of that gateway group (``groups`` is ``partition.groups``).
    """
    kind, name = parse_placement(placement)
    m = min(max(int(round(float(ratio) * num_clients)), 0), num_clients)
    if kind == "group":
        if not groups:
            raise ValueError(f"attack.placement {placement!r} needs partition.groups")
        members = group_members(groups, num_clients)
        if name not in members:
            raise ValueError(f"attack.placement {placement!r}: no group {name!r}; groups are {sorted(members)}")
        if m > len(members[name]):
            raise ValueError(
                f"attack.placement {placement!r}: {m} attackers (malicious_ratio {ratio}) exceed the "
                f"{len(members[name])} clients of group {name!r}"
            )
        return members[name][:m]
    if m == 0:
        return []
    rng = np.random.default_rng(int(seed) + 7919)
    return sorted(int(c) for c in rng.choice(num_clients, size=m, replace=False))


def client_seed(seed: int, round_idx: int, client_id: int) -> int:
    """RNG seed of client ``client_id`` in round ``round_idx`` (batch order, dropout, ...)."""
    return int(seed) + 100_003 * int(round_idx) + 1_009 * int(client_id)


# -- evaluation -------------------------------------------------------------------------------


def monitor_score(metrics: dict[str, Any], monitor: str) -> float | None:
    if monitor == "val_macro_f1":
        return metrics.get("macro_f1")
    if monitor == "val_macro_f1_clean":
        return (metrics.get("clean") or metrics).get("macro_f1")
    if monitor == "val_loss":
        loss = metrics.get("loss")
        return None if loss is None else -float(loss)
    raise ValueError(f"unknown eval.monitor {monitor!r}; expected one of {VALID_MONITORS}")


def stratified_subset(ids: np.ndarray, labels: np.ndarray, cap: int | None, seed: int) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64)
    if not cap or ids.shape[0] <= int(cap):
        return ids
    part = subsample_partition(ClientPartition([ids], "eval", None, seed), labels, int(cap), seed)
    return part.client_flow_ids[0]


class Evaluator:
    """Experiment-only grading of a global parameter vector on the full graph."""

    def __init__(
        self, cfg: dict[str, Any], ctx: ExperimentContext, adapter: ModelAdapter, model: torch.nn.Module, layout: ParamLayout
    ) -> None:
        self.ctx = ctx
        self.adapter = adapter
        self.model = model
        self.layout = layout
        self.families = cfg["eval"].get("families") or {}

    def evaluate(self, vec: torch.Tensor, flow_ids: np.ndarray) -> dict[str, Any]:
        self.model.load_state_dict(self.layout.unflatten(vec))
        gids, logits = self.adapter.predict(self.model, np.asarray(flow_ids, dtype=np.int64))
        return evaluate_predictions(
            gids,
            logits,
            self.ctx.labels,
            self.ctx.label_names,
            clean_labels=self.ctx.clean_labels,
            families=self.families,
        )


def _rss_gb() -> float | None:
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil is an optional dependency
        return None
    return psutil.Process().memory_info().rss / 1024**3


# -- setup ------------------------------------------------------------------------------------


@dataclass
class _Run:
    """Everything a mode runner needs, built once by :func:`run_experiment`."""

    cfg: dict[str, Any]
    mode: str
    out: Path
    device: torch.device
    ctx: ExperimentContext
    adapter: ModelAdapter
    partition: ClientPartition
    clients: list[ClientState]
    layout: ParamLayout
    init_vec: torch.Tensor
    trainer: LocalTrainer
    evaluator: Evaluator
    val_ids: np.ndarray
    attack: Attack
    server: FLServer | None  # None in local_only mode


def _check_config(cfg: dict[str, Any], context: ExperimentContext | None) -> str:
    """Reject a bad config before any work is done; returns the run mode."""
    mode = str(cfg["federated"].get("mode", "fl")).lower()
    if mode not in VALID_MODES:
        raise ValueError(f"unknown federated.mode {mode!r}; expected one of {VALID_MODES}")
    monitor = str(cfg["eval"].get("monitor", "val_macro_f1"))
    if monitor not in VALID_MONITORS:
        raise ValueError(f"unknown eval.monitor {monitor!r}; expected one of {VALID_MONITORS}")
    bad_keys = unknown_keys(cfg)
    if bad_keys:
        # A typo such as attack.malicous_ratio would silently run a clean baseline.
        raise ValueError(f"unknown config keys (typo?): {bad_keys}")
    if context is not None and context.data_key != data_key(cfg):
        raise ValueError("experiment context was loaded for a different data section")
    dtype = str(cfg["output"].get("save_deltas_dtype", "float16"))
    if dtype not in ("float16", "float32"):
        raise ValueError(f"output.save_deltas_dtype must be 'float16' or 'float32', got {dtype!r}")
    rounds = int(cfg["federated"]["rounds"])
    if rounds < 1:
        raise ValueError(f"federated.rounds must be >= 1, got {rounds}")
    round_lr(cfg["client"], 1, rounds)  # rejects an unknown client.lr_schedule
    check_partition_config(cfg["partition"])  # unknown scheme, bad explicit groups / families / dominance
    attack_cfg = cfg["attack"]
    placement = attack_cfg.get("placement", "random")
    parse_placement(placement)
    if mode == "fl" and str(attack_cfg.get("name", "none")).lower() != "none":
        # Dry run: an unknown group or too many attackers for it must fail before training.
        select_malicious(
            int(cfg["partition"]["num_clients"]),
            float(attack_cfg.get("malicious_ratio", 0.0)),
            0,
            placement=placement,
            groups=cfg["partition"].get("groups"),
        )
    return mode


def client_cache_tag(model_name: str, partition: ClientPartition, cfg: dict[str, Any]) -> str:
    """Identifies the cached per-client model data.

    Covers the model name, every client's flow ids and the config the data is built from:
    ``model`` and ``sampler`` (``make_sampler`` reads ``sampler.*`` and ``model.num_layers``)
    and ``data.standardize_flow_features``. Anything else in ``data`` is fixed by the context.
    """
    digest = hashlib.sha1()
    for ids in partition.client_flow_ids:
        arr = np.ascontiguousarray(ids, dtype=np.int64)
        digest.update(np.int64(arr.shape[0]).tobytes())
        digest.update(arr.tobytes())
    built_from = {
        "model": cfg["model"],
        "sampler": cfg["sampler"],
        "standardize_flow_features": cfg["data"].get("standardize_flow_features", True),
    }
    digest.update(json.dumps(to_jsonable(built_from), sort_keys=True).encode("utf-8"))
    return f"{model_name}:{digest.hexdigest()}"


def build_clients(
    cfg: dict[str, Any], ctx: ExperimentContext, adapter: ModelAdapter, partition: ClientPartition
) -> list[ClientState]:
    """Per-client state; model data is cached on ``ctx`` while the partition and model stay put."""
    seed = int(cfg["federated"].get("seed", 42))
    use_cache = bool(cfg["federated"].get("cache_clients", True))
    if use_cache:
        tag = client_cache_tag(adapter.name, partition, cfg)
        if ctx.cache_tag != tag:
            ctx.client_cache.clear()
            ctx.cache_tag = tag
    client_cfg = cfg["client"]
    balanced = str(client_cfg.get("class_weight", "balanced")).lower() == "balanced"
    states: list[ClientState] = []
    for cid, ids in enumerate(partition.client_flow_ids):
        ids = np.asarray(ids, dtype=np.int64)
        labels = ctx.labels[ids]
        data = ctx.client_cache.get(cid) if use_cache else None
        if data is None:
            data = adapter.client_data(cid, ids, seed + cid)
            if use_cache:
                ctx.client_cache[cid] = data
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
        states.append(ClientState(cid, ids, labels, data, weights, adapter.client_eacs()))
    return states


def save_checkpoint(
    path: Path,
    run: _Run,
    vec: torch.Tensor,
    round_idx: int,
    val_metrics: dict[str, Any] | None,
    test_metrics: dict[str, Any] | None,
    extra: dict[str, Any],
) -> None:
    torch.save(
        {
            **run.adapter.checkpoint_meta(),
            "model_state_dict": run.layout.unflatten(vec),
            "config": to_jsonable(run.cfg),
            "num_classes": int(run.ctx.num_classes),
            "label_names": dict(run.ctx.label_names),
            "epoch": int(round_idx),
            "val_metrics": to_jsonable(compact_metrics(val_metrics)),
            "test_metrics": to_jsonable(compact_metrics(test_metrics)) if test_metrics else None,
            "fl": to_jsonable(extra),
        },
        path,
    )


# -- federated mode ---------------------------------------------------------------------------


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


def _summary_header(run: _Run) -> dict[str, Any]:
    cfg = run.cfg
    return {
        "experiment": cfg["experiment"].get("name"),
        "labels": cfg["experiment"].get("labels") or {},
        "mode": run.mode,
        "model": run.adapter.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(run.device),
        "num_clients": len(run.clients),
        "client_sizes": run.partition.sizes(),
        "num_params": run.layout.numel,
        "rounds": int(cfg["federated"]["rounds"]),
    }


def _run_federated(run: _Run) -> dict[str, Any]:
    cfg, ctx, layout, server, attack = run.cfg, run.ctx, run.layout, run.server, run.attack
    assert server is not None
    fed = cfg["federated"]
    rounds = int(fed["rounds"])
    seed = int(fed.get("seed", 42))
    attack_cfg = cfg["attack"]
    knowledge = str(attack_cfg.get("knowledge", "omniscient"))
    start_round = int(attack_cfg.get("start_round", 1))
    malicious = (
        []
        if attack.name == "none"
        else select_malicious(
            len(run.clients),
            float(attack_cfg.get("malicious_ratio", 0.0)),
            seed,
            placement=str(attack_cfg.get("placement", "random")),
            groups=cfg["partition"].get("groups"),
        )
    )
    mal_set = set(malicious)
    rounds_path = run.out / "rounds.jsonl"
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
    if run.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(run.device)
    save_deltas = bool(cfg["output"].get("save_deltas", False))
    deltas_dir = run.out / "deltas"
    deltas_dtype = str(cfg["output"].get("save_deltas_dtype", "float16"))
    if save_deltas:
        cost["saved_delta_dtypes"] = {}
        write_delta_meta(
            deltas_dir,
            layout,
            weighting=server.weighting,
            server_lr=server.server_lr,
            malicious_clients=malicious,
            rounds=rounds,
            assumed_malicious_ratio=float(cfg["defense"].get("assumed_malicious_ratio", 0.2)),
            seed=seed,
        )
    run_start = time.perf_counter()
    for r in range(1, rounds + 1):
        round_start = time.perf_counter()
        lr = round_lr(cfg["client"], r, rounds)
        participants = select_participants(len(run.clients), float(fed.get("participation", 1.0)), seed, r)
        attack_on = attack.name != "none" and r >= start_round
        need_honest = attack.needs_honest or knowledge == "partial"
        updates: dict[int, LocalUpdate] = {}
        t_train = time.perf_counter()
        for cid in participants:
            if attack_on and cid in mal_set and not need_honest:
                continue  # omniscient attacker that crafts without training
            updates[cid] = run.trainer.train(
                run.clients[cid], server.global_vec, r, lr, seed=client_seed(seed, r, cid)
            )
        train_seconds = time.perf_counter() - t_train
        deltas = {cid: upd.delta for cid, upd in updates.items()}
        # Before start_round the designated attackers are honest clients: nothing to craft or grade.
        active = mal_set if attack_on else set()
        mal_now = [c for c in participants if c in active]
        if mal_now:
            deltas.update(_craft(attack, knowledge, updates, participants, mal_set, mal_now, layout.numel, run.device))
        messages = [ClientMessage(c, r, deltas[c], int(run.clients[c].flow_ids.shape[0])) for c in participants]
        report = server.step(messages)
        if save_deltas:
            ids, saved, samples = server.last_inputs or ([], torch.zeros((0, layout.numel)), torch.zeros(0))
            cost["saved_delta_dtypes"][str(r)] = save_round_deltas(
                deltas_dir, r, ids, saved.detach().cpu(), samples.detach().cpu(), sorted(active), deltas_dtype
            )
        det = detection_counts(report.decision_map(), active)
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
            "decisions": [asdict(d) for d in report.decisions],  # includes each client's trust
            "agg_info": report.info,
            "detection": det,
            "client_metrics": {c: {**u.metrics, "steps": u.steps, "seconds": u.seconds} for c, u in updates.items()},
            "train_seconds": train_seconds,
            "server_seconds": report.seconds,
        }
        if r % every == 0 or r == rounds:
            t_eval = time.perf_counter()
            last_val = run.evaluator.evaluate(server.global_vec, run.val_ids)
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
    test_final = run.evaluator.evaluate(final_vec, ctx.test_ids)
    if best["vec"] is None:
        best = {"score": None, "round": rounds, "vec": final_vec, "val": last_val}
    test_best = test_final if best["round"] == rounds else run.evaluator.evaluate(best["vec"], ctx.test_ids)
    if run.device.type == "cuda":
        cost["peak_vram_gb"] = torch.cuda.max_memory_allocated(run.device) / 1024**3
    cost["wall_seconds"] = time.perf_counter() - run_start
    cost["mean_round_seconds"] = cost["wall_seconds"] / max(rounds, 1)
    extra = {
        "mode": run.mode,
        "malicious_clients": malicious,
        "attack": attack.name,
        "aggregator": server.aggregator.name,
    }
    if cfg["output"].get("save_checkpoints", True):
        save_checkpoint(run.out / "global_final.pt", run, final_vec, rounds, last_val, test_final, extra)
        save_checkpoint(run.out / "global_best.pt", run, best["vec"], best["round"], best["val"], test_best, extra)
    return {
        **_summary_header(run),
        "malicious_clients": malicious,
        "attack": attack.name,
        "aggregator": server.aggregator.name,
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


# -- local_only mode --------------------------------------------------------------------------


def _metric_stats(per_client: list[dict[str, Any]], split: str, clean: bool = False) -> dict[str, float | None]:
    values = []
    for row in per_client:
        m = row[split].get("clean") if clean else row[split]
        if m and m.get("macro_f1") is not None:
            values.append(float(m["macro_f1"]))
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    arr = np.asarray(values)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}


def _run_local_only(run: _Run) -> dict[str, Any]:
    cfg, ctx = run.cfg, run.ctx
    rounds = int(cfg["federated"]["rounds"])
    seed = int(cfg["federated"].get("seed", 42))
    rounds_path = run.out / "rounds.jsonl"
    if rounds_path.exists():
        rounds_path.unlink()
    per_client: list[dict[str, Any]] = []
    start = time.perf_counter()
    for state in run.clients:
        vec = run.init_vec.clone()
        for r in range(1, rounds + 1):
            update = run.trainer.train(
                state, vec, r, round_lr(cfg["client"], r, rounds), seed=client_seed(seed, r, state.client_id)
            )
            vec = vec + update.delta
        val = run.evaluator.evaluate(vec, run.val_ids)
        test = run.evaluator.evaluate(vec, ctx.test_ids)
        per_client.append({"client_id": state.client_id, "num_flows": int(state.flow_ids.shape[0]), "val": val, "test": test})
        append_jsonl(rounds_path, {"client_id": state.client_id, "val": compact_metrics(val), "test": compact_metrics(test)})
        print(f"[local_only] client {state.client_id} test_macro_f1={test.get('macro_f1')}", flush=True)
    return {
        **_summary_header(run),
        "malicious_clients": [],
        "attack": "none",
        "aggregator": None,
        "per_client": per_client,
        "aggregate": {
            "val_macro_f1": _metric_stats(per_client, "val"),
            "val_macro_f1_clean": _metric_stats(per_client, "val", clean=True),
            "test_macro_f1": _metric_stats(per_client, "test"),
            "test_macro_f1_clean": _metric_stats(per_client, "test", clean=True),
        },
        "cost": {"wall_seconds": time.perf_counter() - start, "peak_rss_gb": _rss_gb()},
        "config": cfg,
    }


# -- entry point ------------------------------------------------------------------------------


def run_experiment(
    cfg: dict[str, Any], *, out_dir: str | Path | None = None, context: ExperimentContext | None = None
) -> dict[str, Any]:
    """Run one experiment and return its summary (also written to ``fl_summary.json``)."""
    mode = _check_config(cfg, context)
    if mode == "centralized":
        cfg = apply_overrides(cfg, CENTRALIZED_OVERRIDES)
    elif mode == "local_only":
        cfg = apply_overrides(cfg, LOCAL_ONLY_OVERRIDES)
    if out_dir is not None:
        cfg = apply_overrides(cfg, {"output.dir": str(out_dir)})
    seed = int(cfg["federated"].get("seed", 42))
    defense = cfg["defense"]
    # Built up front only to reject bad names / params before any work is done.
    attack = build_attack(cfg["attack"])
    aggregator = build_defense(
        defense["aggregator"],
        # The defender's fixed assumption about attackers; never derived from the attack config.
        assumed_malicious_ratio=float(defense.get("assumed_malicious_ratio", 0.2)),
        seed=seed,
    )

    set_seed(seed)
    device = resolve_device(str(cfg.get("device", "auto")))
    ctx = context if context is not None else load_context(cfg)

    partition = make_partition(
        ctx.train_ids, ctx.labels, cfg["partition"], label_names=ctx.label_names, order=ctx.flow_order
    )
    if mode == "centralized":
        # Same (possibly subsampled) training flows as the FL clients, held by one silo.
        partition = ClientPartition([partition.all_flow_ids()], "centralized", None, seed)
    # Round-0 protocol message: the pooled flow-feature moments every client standardizes with.
    flow_stats = federated_flow_stats(ctx.artifact.node_features["flow"], partition.client_flow_ids)
    adapter = build_adapter(cfg, ctx, flow_stats, device)  # rejects an unknown model.name
    torch.manual_seed(seed)  # initial weights depend on the seed alone, not on earlier RNG draws
    model = adapter.build_model()
    layout = ParamLayout.from_model(model)
    init_vec = layout.flatten(model.state_dict())
    trainer = LocalTrainer(model, layout, adapter, device, cfg["client"])
    server = None
    if mode != "local_only":
        server = FLServer(
            global_vec=init_vec,
            aggregator=aggregator,
            layout=layout,
            weighting=str(defense.get("weighting", "samples")),
            server_lr=float(cfg["federated"].get("server_lr", 1.0)),
            device=device,
        )

    out = Path(cfg["output"]["dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.resolved.yaml").write_text(yaml.safe_dump(to_jsonable(cfg), sort_keys=False), encoding="utf-8")
    write_json(
        out / "partition.json",
        {
            **partition.to_json(),
            "sizes": partition.sizes(),
            "class_histogram": partition.class_histogram(ctx.labels, ctx.num_classes),
        },
    )

    run = _Run(
        cfg=cfg,
        mode=mode,
        out=out,
        device=device,
        ctx=ctx,
        adapter=adapter,
        partition=partition,
        clients=build_clients(cfg, ctx, adapter, partition),
        layout=layout,
        init_vec=init_vec,
        trainer=trainer,
        evaluator=Evaluator(cfg, ctx, adapter, model, layout),
        val_ids=stratified_subset(ctx.val_ids, ctx.labels, cfg["eval"].get("max_val_flows"), seed),
        attack=attack,
        server=server,
    )
    summary = _run_local_only(run) if mode == "local_only" else _run_federated(run)
    write_json(out / "fl_summary.json", summary)
    return to_jsonable(summary)
