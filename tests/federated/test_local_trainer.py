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
    # Plain shuffling visits every flow once per epoch, so the fixture's only two suspects
    # (flows 1 and 4) are seen for sure; stratified sampling draws with replacement and,
    # for seed 0, happens to miss both.
    trainer, layout, state, adapter, _ = _setup(
        smoke_cfg, np.arange(30), **{"client.stratified_batch_sampling": False}
    )
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
