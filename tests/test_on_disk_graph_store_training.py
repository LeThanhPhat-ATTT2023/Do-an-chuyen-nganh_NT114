from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from graphslm_ids.offline_path.training.hetero_graph_artifact import load_graph_store_artifact
from graphslm_ids.offline_path.training.on_disk_graph_store import (
    OnDiskHeteroGraphStore,
    convert_npz_to_on_disk_graph_store,
)
from graphslm_ids.offline_path.training.train_hgt_flow_classifier import train_neighbor_sampling


def _write_tiny_npz(tmp_path):
    graph_npz = tmp_path / "tiny_graph.npz"
    graph_meta = tmp_path / "tiny_graph.meta.json"
    flow_x = np.asarray(
        [
            [1, 10, 0, 1000, 80, 1],
            [1, 11, 0, 1001, 80, 1],
            [1, 12, 0, 1002, 80, 1],
            [1, 20, 0, 2000, 443, 1],
            [1, 21, 0, 2001, 443, 1],
            [1, 22, 0, 2002, 443, 1],
        ],
        dtype=np.float32,
    )
    flow_y = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    packet_semantic_x = np.asarray(
        [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=np.float32,
    )
    contain = np.vstack([np.arange(6, dtype=np.int64), np.arange(6, dtype=np.int64)])
    packet_tech = np.vstack(
        [
            np.arange(6, dtype=np.int64),
            np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64),
        ]
    )
    flow_tech = packet_tech.copy()
    np.savez_compressed(
        graph_npz,
        flow_x=flow_x,
        flow_y=flow_y,
        packet_x=np.zeros((6, 256), dtype=np.uint8),
        packet_semantic_x=packet_semantic_x,
        technique_x=np.eye(2, 4, dtype=np.float32),
        contain_edge_index=contain,
        link_edge_index=np.empty((2, 0), dtype=np.int64),
        link_edge_attr=np.empty((0, 1), dtype=np.float32),
        packet_technique_edge_index=packet_tech,
        packet_technique_edge_attr=np.full((6, 1), 0.9, dtype=np.float32),
        flow_technique_edge_index=flow_tech,
        flow_technique_edge_attr=np.full((6, 1), 0.95, dtype=np.float32),
        technique_tactic_edge_index=np.asarray([[0, 1], [0, 1]], dtype=np.int64),
        technique_tactic_edge_attr=np.ones((2, 1), dtype=np.float32),
    )
    graph_meta.write_text(
        json.dumps(
            {
                "num_tactics": 2,
                "label_mapping": {"benign": 0, "malicious": 1},
                "flow_feature_names": [
                    "packet_count",
                    "total_payload_bytes",
                    "duration_seconds",
                    "src_port",
                    "dst_port",
                    "protocol_id",
                ],
                "technique_id_to_idx": {"T1001": 0, "T1002": 1},
                "tactic_shortname_to_idx": {"initial-access": 0, "execution": 1},
            }
        ),
        encoding="utf-8",
    )
    return graph_npz, graph_meta


def test_npz_to_mmap_csr_graph_store_round_trip(tmp_path) -> None:
    graph_npz, graph_meta = _write_tiny_npz(tmp_path)
    root = tmp_path / "graph_store_v1"

    manifest = convert_npz_to_on_disk_graph_store(
        graph_npz=graph_npz,
        graph_meta_json=graph_meta,
        output_root=root,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=7,
    )
    store = OnDiskHeteroGraphStore(root)
    artifact = load_graph_store_artifact(root, add_reverse_edges=False)

    assert manifest["layout"] == "numpy_memmap_csr"
    assert store.flow_shard_index is not None
    assert store.packet_shard_index is not None
    assert np.asarray(store.flow_shard_index).tolist() == [0, 0, 0, 0, 0, 0]
    assert artifact.node_features["flow"].shape == (6, 6)
    assert artifact.node_features["packet"].shape == (6, 4)
    assert artifact.node_features["technique"].shape == (2, 4)
    assert artifact.node_features["tactic"].reshape(-1).tolist() == [0, 1]
    assert artifact.flow_y.tolist() == [0, 0, 0, 1, 1, 1]
    indptr, neighbors, _ = store.out_neighbors(
        ("flow", "contains", "packet"),
        np.asarray([0, 3], dtype=np.int64),
    )
    assert indptr.tolist() == [0, 1, 2]
    assert neighbors.tolist() == [0, 3]
    assert manifest["class_counts"] == [1, 1]
    assert manifest["shards"]["flow"] == [{"shard_id": 0, "range": [0, 6]}]


def test_neighbor_sampling_training_smoke_on_npz(tmp_path) -> None:
    graph_npz, graph_meta = _write_tiny_npz(tmp_path)
    config = {
        "data": {
            "source": "npz",
            "graph_npz": str(graph_npz),
            "graph_meta_json": str(graph_meta),
            "graph_store_root": str(tmp_path / "unused"),
            "read_sealed_only": True,
            "packet_feature": "semantic",
            "add_reverse_edges": True,
            "standardize_flow_features": False,
            "use_semantic_edge_weights": True,
        },
        "model": {
            "hidden_dim": 8,
            "num_layers": 1,
            "num_heads": 2,
            "dropout": 0.0,
            "ffn_multiplier": 1,
        },
        "train": {
            "output_dir": str(tmp_path / "out"),
            "epochs": 1,
            "batch_mode": "neighbor_sampling",
            "batch_seed_flows": 2,
            "grad_accum_steps": 1,
            "lr": 0.001,
            "weight_decay": 0.0,
            "val_ratio": 0.2,
            "test_ratio": 0.2,
            "patience": 3,
            "class_weight": "balanced",
            "seed": 123,
            "device": "cpu",
            "monitor": "val_macro_f1",
            "log_every": 1,
            "amp": False,
            "activation_checkpointing": False,
        },
        "sampler": {
            "hops": 1,
            "fanouts": {
                "flow__contains__packet": 2,
                "packet__matches_technique__technique": 2,
                "flow__matches_technique__technique": 2,
                "technique__belongs_to_tactic__tactic": 1,
            },
            "reverse_fanouts": {"rev_contains": 0, "rev_matches_technique": 0},
            "always_include_all_tactics": True,
            "always_include_all_techniques": True,
        },
        "dataloader": {"num_workers": 0, "prefetch_factor": 2, "pin_memory": False},
    }

    train_neighbor_sampling(config, seed=123, device=torch.device("cpu"))

    assert (tmp_path / "out" / "hgt_flow_best.pt").exists()
    assert (tmp_path / "out" / "training_summary.json").exists()
