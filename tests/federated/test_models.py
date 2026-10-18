import numpy as np
import pytest
import torch

from graphslm_ids.offline.federated.config import DEFAULT_FL_CONFIG
from graphslm_ids.offline.federated.context import federated_flow_stats, load_context
from graphslm_ids.offline.federated.local_trainer import ClientState, LocalTrainer, local_class_weights
from graphslm_ids.offline.federated.models import (
    ByteCNN1D,
    FlowMLP,
    TensorClientData,
    build_adapter,
    byte_columns,
    flow_byte_matrix,
    flow_packet_csr,
    rebuild_model,
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


def test_byte_cnn_rejects_input_too_short_for_three_pools():
    with pytest.raises(ValueError):
        ByteCNN1D(14, 3, channels=(4, 8, 8), num_groups=2)
    assert ByteCNN1D(15, 3, channels=(4, 8, 8), num_groups=2)(torch.randn(2, 1, 15)).shape == (2, 3)


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


def test_flow_packet_csr_orders_each_flow_by_packet_id():
    ptr, idx = flow_packet_csr(np.array([1, 0, 1, 0, -1, 1]), num_flows=2)
    assert ptr.tolist() == [0, 2, 5] and idx.tolist() == [1, 3, 0, 2, 5]
    assert flow_byte_matrix(np.array([], dtype=np.int64), ptr, idx, np.zeros((6, 2), np.float16), np.array([0]), 2).shape == (0, 2)


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


def _adapter(smoke_cfg, name, ids, **overrides):
    cfg = smoke_cfg(name, **{"model.name": name, "model.cnn.num_groups": 2, **overrides})
    ctx = load_context(cfg)
    stats = federated_flow_stats(ctx.artifact.node_features["flow"], [ids])
    return cfg, ctx, build_adapter(cfg, ctx, stats, torch.device("cpu")), stats


def test_mlp_client_data_is_standardized_float32(smoke_cfg):
    ids = np.arange(20, dtype=np.int64)
    _, ctx, adapter, stats = _adapter(smoke_cfg, "mlp", ids)
    data = adapter.client_data(0, ids, seed=0)
    assert isinstance(data, TensorClientData) and data.x.dtype == torch.float32 and data.y.dtype == torch.int64
    assert tuple(data.x.shape) == (20, 6) and data.y.tolist() == ctx.labels[ids].tolist()
    expected = (ctx.artifact.node_features["flow"][ids] - np.asarray(stats["mean"])) / np.asarray(stats["std"])
    assert np.allclose(data.x.numpy(), expected, atol=1e-5)
    # Pooled over exactly these flows, the standardized columns have mean ~0.
    assert np.allclose(data.x.numpy().mean(axis=0), 0.0, atol=1e-4)


def test_cnn_client_data_is_float16_bytes_in_first_packets_order(smoke_cfg):
    ids = np.array([5, 3, 9], dtype=np.int64)
    _, ctx, adapter, _ = _adapter(smoke_cfg, "cnn", ids)
    data = adapter.client_data(0, ids, seed=0)
    assert data.x.dtype == torch.float16 and tuple(data.x.shape) == (3, 48) and data.y.dtype == torch.int64
    packet_x = np.asarray(ctx.artifact.node_features["packet"], dtype=np.float32)
    # Tiny fixture: flow f owns packets 2f, 2f+1; byte columns are the 6 obyte_* columns 2..7.
    for row, f in enumerate(ids.tolist()):
        want = np.zeros((8, 6), dtype=np.float32)
        want[:2] = packet_x[[2 * f, 2 * f + 1]][:, 2:8]
        assert np.allclose(data.x[row].float().numpy(), want.reshape(-1), atol=1e-2)
    assert adapter.checkpoint_meta()["input_len"] == 48


def test_baseline_train_batches_cover_every_flow_with_ids(smoke_cfg):
    ids = np.arange(20, dtype=np.int64)
    cfg, ctx, adapter, _ = _adapter(smoke_cfg, "mlp", ids)
    data = adapter.client_data(0, ids, seed=0)
    batches = list(adapter.train_batches(data, ids, ctx.labels[ids], seed=3))
    assert [len(b[0]) for b in batches] == [8, 8, 4]  # client.batch_seed_flows = 8
    seen = torch.cat([b[2] for b in batches]).tolist()
    assert sorted(seen) == ids.tolist()
    again = torch.cat([b[2] for b in adapter.train_batches(data, ids, ctx.labels[ids], seed=3)]).tolist()
    assert again == seen


@pytest.mark.parametrize("name", ["mlp", "cnn"])
def test_rebuild_model_from_checkpoint_meta(smoke_cfg, name):
    ids = np.arange(20, dtype=np.int64)
    cfg, _, adapter, _ = _adapter(smoke_cfg, name, ids)
    model = adapter.build_model()
    ckpt = {**adapter.checkpoint_meta(), "model_state_dict": model.state_dict(), "config": cfg, "num_classes": 3}
    rebuilt = rebuild_model(ckpt)
    assert type(rebuilt) is type(model)
    rebuilt.load_state_dict(ckpt["model_state_dict"])


def test_unknown_model_name_is_rejected_before_training(smoke_cfg):
    ids = np.arange(5, dtype=np.int64)
    cfg = smoke_cfg("bogus", **{"model.name": "bogus"})
    ctx = load_context(cfg)
    with pytest.raises(ValueError):
        build_adapter(cfg, ctx, federated_flow_stats(ctx.artifact.node_features["flow"], [ids]), torch.device("cpu"))
    with pytest.raises(ValueError):
        rebuild_model({"model_name": "bogus"})


def test_cnn_adapter_without_packet_feature_names_fails(smoke_cfg):
    ids = np.arange(5, dtype=np.int64)
    cfg = smoke_cfg("nonames", **{"model.name": "cnn"})
    ctx = load_context(cfg)
    ctx.artifact.metadata.pop("packet_feature_names")
    with pytest.raises(ValueError):
        build_adapter(cfg, ctx, federated_flow_stats(ctx.artifact.node_features["flow"], [ids]), torch.device("cpu"))


def test_default_config_carries_baseline_model_sections():
    model = DEFAULT_FL_CONFIG["model"]
    assert model["mlp"] == {"hidden_dims": [256, 128, 64], "dropout": 0.2}
    assert model["cnn"] == {
        "num_packets": 8,
        "byte_prefix": "obyte_",
        "channels": [32, 64, 128],
        "kernel_size": 7,
        "num_groups": 8,
        "dropout": 0.2,
    }
