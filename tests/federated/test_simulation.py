import json

import pytest
import torch

from graphslm_ids.offline.federated import simulation
from graphslm_ids.offline.federated.config import apply_overrides
from graphslm_ids.offline.federated.local_trainer import LocalTrainer
from graphslm_ids.offline.federated.models import rebuild_model
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


def test_fl_smoke_sign_flip_fedavg(smoke_cfg, tmp_path):
    cfg = smoke_cfg("fl", **{"attack.name": "sign_flip", "attack.malicious_ratio": 0.34})
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
    assert ckpt["model_name"] == "hgt" and ckpt["label_names"][1] == "XSS" and ckpt["flow_feature_stats"]["mean"]
    rebuild_model(ckpt).load_state_dict(ckpt["model_state_dict"])
    assert summary["model"] == "hgt"


def test_fl_run_records_trust_agg_info_and_cost(smoke_cfg, tmp_path):
    summary = run_experiment(smoke_cfg("rec", **{"attack.name": "sign_flip", "attack.malicious_ratio": 0.34}))
    first = _records(tmp_path / "rec" / "rounds.jsonl")[0]
    assert {"client_id", "decision", "trust", "weight", "norm"} <= set(first["decisions"][0])
    assert "agg_info" in first and "screen_features" not in first and "val" in first
    part = json.loads((tmp_path / "rec" / "partition.json").read_text(encoding="utf-8"))
    assert part["scheme"] == "iid" and len(part["sizes"]) == 3 and len(part["class_histogram"]) == 3
    assert summary["client_sizes"] == part["sizes"] and summary["aggregator"] == "fedavg"
    assert "screen" not in summary
    assert summary["best"]["test"] and summary["best_round"] in (1, 2) and len(summary["history"]) == 2
    cost = summary["cost"]
    assert cost["bytes_up"] > 0 and cost["peak_rss_gb"] > 0 and cost["wall_seconds"] > 0


def test_fl_smoke_omniscient_alie_skips_attacker_training(smoke_cfg, tmp_path):
    cfg = smoke_cfg("alie", **{"attack.name": "alie", "attack.malicious_ratio": 0.34})
    summary = run_experiment(cfg)
    first = _records(tmp_path / "alie" / "rounds.jsonl")[0]
    assert str(summary["malicious_clients"][0]) not in first["client_metrics"]


def test_partial_participation_selects_a_subset_each_round(smoke_cfg, tmp_path):
    run_experiment(smoke_cfg("part", **{"federated.participation": 0.67, "federated.rounds": 3}))
    records = _records(tmp_path / "part" / "rounds.jsonl")
    assert [len(r["participants"]) for r in records] == [2, 2, 2]
    assert all(len(r["decisions"]) == 2 for r in records)


def test_client_rng_seed_follows_the_round_client_formula(smoke_cfg, monkeypatch):
    calls = []
    original = LocalTrainer.train

    def spy(self, state, global_vec, round_idx, lr, seed):
        calls.append((round_idx, state.client_id, seed))
        return original(self, state, global_vec, round_idx, lr, seed)

    monkeypatch.setattr(LocalTrainer, "train", spy)
    run_experiment(smoke_cfg("seed", **{"federated.seed": 5}))
    assert sorted((r, k) for r, k, _ in calls) == [(r, k) for r in (1, 2) for k in range(3)]
    assert all(seed == 5 + 100_003 * r + 1_009 * k for r, k, seed in calls)


def test_assumed_malicious_ratio_comes_from_defense_config_only(smoke_cfg, monkeypatch):
    seen = []
    original = simulation.build_aggregator

    def spy(cfg, *, assumed_malicious_ratio, seed):
        seen.append(assumed_malicious_ratio)
        return original(cfg, assumed_malicious_ratio=assumed_malicious_ratio, seed=seed)

    monkeypatch.setattr(simulation, "build_aggregator", spy)
    run_experiment(smoke_cfg("amr1", **{"attack.name": "sign_flip", "attack.malicious_ratio": 0.34}))
    run_experiment(smoke_cfg("amr2", **{"defense.assumed_malicious_ratio": 0.1, "federated.rounds": 1}))
    assert seen == [0.2, 0.1]


def test_save_checkpoints_can_be_disabled(smoke_cfg, tmp_path):
    run_experiment(smoke_cfg("nockpt", **{"output.save_checkpoints": False, "federated.rounds": 1}))
    out = tmp_path / "nockpt"
    assert (out / "fl_summary.json").exists() and not list(out.glob("*.pt"))


def test_centralized_mode_trains_on_union_of_client_flows(smoke_cfg, tmp_path):
    summary = run_experiment(
        smoke_cfg("central", **{"federated.mode": "centralized", "attack.name": "sign_flip", "attack.malicious_ratio": 0.34})
    )
    assert summary["num_clients"] == 1 and summary["malicious_clients"] == [] and summary["attack"] == "none"
    assert summary["aggregator"] == "fedavg"
    part = json.loads((tmp_path / "central" / "partition.json").read_text(encoding="utf-8"))
    assert part["scheme"] == "centralized" and part["sizes"] == [32]
    assert summary["config"]["client"]["optimizer_state"] == "persist"
    assert summary["config"]["federated"]["participation"] == 1.0


def test_local_only_mode_reports_every_client(smoke_cfg, tmp_path):
    summary = run_experiment(smoke_cfg("local", **{"federated.mode": "local_only"}))
    assert len(summary["per_client"]) == 3 and summary["model"] == "hgt"
    assert summary["aggregate"]["test_macro_f1"]["mean"] is not None
    assert (tmp_path / "local" / "fl_summary.json").exists()
    assert len(_records(tmp_path / "local" / "rounds.jsonl")) == 3
    assert summary["config"]["client"]["optimizer_state"] == "persist"


def test_context_reuse_and_validation(smoke_cfg):
    cfg = smoke_cfg("ctx")
    ctx = load_context(cfg)
    run_experiment(cfg, context=ctx)
    assert ctx.client_cache and ctx.cache_tag
    with pytest.raises(ValueError):
        run_experiment(apply_overrides(cfg, {"data.split_protocol": "temporal"}), context=ctx)
    with pytest.raises(ValueError):
        run_experiment(apply_overrides(cfg, {"federated.mode": "bogus"}))
    with pytest.raises(ValueError):
        run_experiment(apply_overrides(cfg, {"eval.monitor": "bogus"}))
    with pytest.raises(ValueError):
        run_experiment(apply_overrides(cfg, {"model.name": "bogus"}))
    with pytest.raises(ValueError, match="unknown config keys"):
        run_experiment(apply_overrides(cfg, {"attack.malicous_ratio": 0.3}))


@pytest.mark.parametrize(
    "override",
    [
        {"attack.name": "bogus"},
        {"attack.knowledge": "bogus"},
        {"defense.aggregator": {"name": "bogus", "params": {}}},
        {"defense.weighting": "bogus"},
        {"client.lr_schedule": "bogus"},
        {"client.optimizer_state": "bogus"},
        {"federated.rounds": 0},
    ],
)
def test_bad_config_fails_before_anything_is_written(smoke_cfg, tmp_path, override):
    with pytest.raises(ValueError):
        run_experiment(smoke_cfg("bad", **override))
    assert not (tmp_path / "bad").exists()


def test_client_cache_is_reused_for_the_same_partition_and_reset_otherwise(smoke_cfg, tmp_path):
    cfg = smoke_cfg("cache")
    ctx = load_context(cfg)
    run_experiment(cfg, context=ctx)
    tag, cached = ctx.cache_tag, dict(ctx.client_cache)
    assert tag.startswith("hgt:") and sorted(cached) == [0, 1, 2]

    attacked = apply_overrides(cfg, {"attack.name": "sign_flip", "attack.malicious_ratio": 0.34})
    run_experiment(attacked, out_dir=tmp_path / "cache_attacked", context=ctx)
    assert ctx.cache_tag == tag and all(ctx.client_cache[c] is cached[c] for c in cached)

    run_experiment(apply_overrides(cfg, {"partition.seed": 1}), out_dir=tmp_path / "cache_reseeded", context=ctx)
    assert ctx.cache_tag != tag and sorted(ctx.client_cache) == [0, 1, 2]

    run_experiment(apply_overrides(cfg, {"federated.mode": "centralized"}), out_dir=tmp_path / "cache_central", context=ctx)
    assert sorted(ctx.client_cache) == [0]


def test_cache_clients_false_leaves_the_context_cache_untouched(smoke_cfg):
    cfg = smoke_cfg("nocache", **{"federated.cache_clients": False, "federated.rounds": 1})
    ctx = load_context(cfg)
    run_experiment(cfg, context=ctx)
    assert not ctx.client_cache and ctx.cache_tag is None


def test_out_dir_argument_overrides_output_dir(smoke_cfg, tmp_path):
    summary = run_experiment(smoke_cfg("cfgdir", **{"federated.rounds": 1}), out_dir=tmp_path / "elsewhere")
    assert (tmp_path / "elsewhere" / "fl_summary.json").exists() and not (tmp_path / "cfgdir").exists()
    assert summary["config"]["output"]["dir"] == str(tmp_path / "elsewhere")


def test_centralized_baseline_is_exactly_one_epoch_per_round(smoke_cfg, tmp_path):
    odd = {"client.local_epochs": 2, "client.max_local_steps": 1, "federated.server_lr": 0.5}
    summary = run_experiment(smoke_cfg("central_odd", **{"federated.mode": "centralized", **odd}))
    plain = run_experiment(smoke_cfg("central_plain", **{"federated.mode": "centralized"}))
    resolved = summary["config"]
    assert resolved["client"]["local_epochs"] == 1 and resolved["client"]["max_local_steps"] is None
    assert resolved["federated"]["server_lr"] == 1.0
    assert plain["config"]["client"]["local_epochs"] == 1
    # The overrides are in force, not just recorded: both runs take the same optimizer steps.
    odd_first = _records(tmp_path / "central_odd" / "rounds.jsonl")[0]["client_metrics"]["0"]
    plain_first = _records(tmp_path / "central_plain" / "rounds.jsonl")[0]["client_metrics"]["0"]
    assert odd_first["steps"] == plain_first["steps"] > 1
    assert summary["final"]["test"]["macro_f1"] == plain["final"]["test"]["macro_f1"]


@pytest.mark.parametrize("override", [{"sampler.hops": 1}, {"model.num_layers": 3}, {"model.hidden_dim": 16}])
def test_client_cache_is_rebuilt_when_model_or_sampler_config_changes(smoke_cfg, tmp_path, override):
    cfg = smoke_cfg("cache_cfg", **{"federated.rounds": 1})
    ctx = load_context(cfg)
    run_experiment(cfg, context=ctx)
    tag, cached = ctx.cache_tag, dict(ctx.client_cache)

    run_experiment(cfg, out_dir=tmp_path / "same", context=ctx)
    assert ctx.cache_tag == tag and all(ctx.client_cache[c] is cached[c] for c in cached)

    run_experiment(apply_overrides(cfg, override), out_dir=tmp_path / "changed", context=ctx)
    assert ctx.cache_tag != tag and sorted(ctx.client_cache) == sorted(cached)
    assert all(ctx.client_cache[c] is not cached[c] for c in cached)


def test_detection_is_graded_only_in_rounds_where_the_attack_is_active(smoke_cfg, tmp_path):
    cfg = smoke_cfg("late", **{"attack.name": "sign_flip", "attack.malicious_ratio": 0.34, "attack.start_round": 2})
    summary = run_experiment(cfg)
    first, second = _records(tmp_path / "late" / "rounds.jsonl")
    assert not first["attack_active"] and first["malicious"] == []
    det1 = first["detection"]
    assert det1["tp"] + det1["fn"] == 0 and det1["fp"] + det1["tn"] == 3
    assert second["attack_active"] and second["malicious"] == summary["malicious_clients"]
    assert len(summary["malicious_clients"]) == 1
    assert summary["detection"]["tp"] + summary["detection"]["fn"] == 1
    assert summary["detection"]["fp"] + summary["detection"]["tn"] == 5
