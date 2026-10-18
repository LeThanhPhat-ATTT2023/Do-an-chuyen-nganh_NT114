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
