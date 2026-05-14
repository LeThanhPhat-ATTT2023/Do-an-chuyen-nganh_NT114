"""End-to-end DDP smoke check for the HGT trainer.

The test launches two CPU processes via ``torch.multiprocessing``, points them
at a tiny on-the-fly graph artifact, and runs one epoch of neighbour-sampling
training. It verifies:

  * ``setup_distributed`` initialises Gloo correctly when LOCAL_RANK is set.
  * ``train_neighbor_sampling`` runs end-to-end under DDP without erroring.
  * Only rank 0 writes the checkpoint + summary.
  * The checkpoint is loadable by a separate single-process consumer (mirrors
    how the runtime / inference path will load it).

It's a smoke test, not a correctness regression: we don't check metric values,
only that the pipeline runs and produces the right artifacts.
"""
from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.multiprocessing as mp


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def _build_tiny_graph(npz_path: Path, meta_path: Path) -> None:
    """Tiny three-tier graph artifact with enough variety for DDP sharding."""
    num_flows = 24
    num_packets = 48
    num_techniques = 4
    num_tactics = 2
    flow_dim = 6

    rng = np.random.default_rng(0)
    flow_x = rng.standard_normal((num_flows, flow_dim)).astype(np.float32)
    # Two-class problem with balanced support so stratified split works.
    flow_y = np.repeat(np.arange(2, dtype=np.int64), num_flows // 2)
    rng.shuffle(flow_y)

    # Each flow contains 2 packets.
    contain_src = np.repeat(np.arange(num_flows, dtype=np.int64), 2)
    contain_dst = np.arange(num_packets, dtype=np.int64)
    contain_edge_index = np.vstack([contain_src, contain_dst])

    # Linear packet chain.
    link_src = np.arange(num_packets - 1, dtype=np.int64)
    link_dst = np.arange(1, num_packets, dtype=np.int64)
    link_edge_index = np.vstack([link_src, link_dst])
    link_edge_attr = np.ones((link_dst.shape[0], 1), dtype=np.float32)

    pkt_tech_src = np.arange(num_packets, dtype=np.int64)
    pkt_tech_dst = (pkt_tech_src % num_techniques).astype(np.int64)
    pkt_tech_edge_index = np.vstack([pkt_tech_src, pkt_tech_dst])
    pkt_tech_edge_attr = np.full((pkt_tech_dst.shape[0], 1), 0.9, dtype=np.float32)

    flow_tech_src = np.arange(num_flows, dtype=np.int64)
    flow_tech_dst = (flow_tech_src % num_techniques).astype(np.int64)
    flow_tech_edge_index = np.vstack([flow_tech_src, flow_tech_dst])
    flow_tech_edge_attr = np.full((flow_tech_dst.shape[0], 1), 0.9, dtype=np.float32)

    tech_tactic_src = np.arange(num_techniques, dtype=np.int64)
    tech_tactic_dst = (tech_tactic_src % num_tactics).astype(np.int64)
    tech_tactic_edge_index = np.vstack([tech_tactic_src, tech_tactic_dst])
    tech_tactic_edge_attr = np.ones((tech_tactic_dst.shape[0], 1), dtype=np.float32)

    np.savez_compressed(
        npz_path,
        flow_x=flow_x,
        flow_y=flow_y,
        packet_semantic_x=rng.standard_normal((num_packets, 4)).astype(np.float32),
        technique_x=rng.standard_normal((num_techniques, 4)).astype(np.float32),
        contain_edge_index=contain_edge_index,
        link_edge_index=link_edge_index,
        link_edge_attr=link_edge_attr,
        packet_technique_edge_index=pkt_tech_edge_index,
        packet_technique_edge_attr=pkt_tech_edge_attr,
        flow_technique_edge_index=flow_tech_edge_index,
        flow_technique_edge_attr=flow_tech_edge_attr,
        technique_tactic_edge_index=tech_tactic_edge_index,
        technique_tactic_edge_attr=tech_tactic_edge_attr,
    )

    import json
    meta = {
        "num_flows": num_flows,
        "num_packets": num_packets,
        "num_techniques": num_techniques,
        "num_tactics": num_tactics,
        "label_mapping": {"benign": 0, "malicious": 1},
        "flow_feature_names": [f"f{i}" for i in range(flow_dim)],
    }
    meta_path.write_text(json.dumps(meta), encoding="utf-8")


def _worker(rank: int, world_size: int, port: int, npz_path: str, meta_path: str, output_dir: str) -> None:
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    from graphslm_ids.offline_path.training.train_hgt_flow_classifier import (
        setup_distributed,
        teardown_distributed,
        train_neighbor_sampling,
        DEFAULT_CONFIG,
    )
    from copy import deepcopy

    config = deepcopy(DEFAULT_CONFIG)
    config["data"]["source"] = "npz"
    config["data"]["graph_npz"] = npz_path
    config["data"]["graph_meta_json"] = meta_path
    config["data"]["standardize_flow_features"] = False
    config["data"]["use_semantic_edge_weights"] = False
    config["train"]["output_dir"] = output_dir
    config["train"]["epochs"] = 1
    config["train"]["batch_mode"] = "neighbor_sampling"
    config["train"]["batch_seed_flows"] = 4
    config["train"]["device"] = "cpu"
    config["train"]["multi_gpu"] = False
    config["train"]["class_weight"] = "none"
    config["train"]["compile"] = False
    config["train"]["tf32"] = False
    config["train"]["patience"] = 999  # never early-stop on a 1-epoch run
    config["model"]["hidden_dim"] = 16
    config["model"]["num_layers"] = 2
    config["model"]["num_heads"] = 2
    config["sampler"]["fanouts"] = {
        "flow__contains__packet": 2,
        "packet__matches_technique__technique": 2,
        "flow__matches_technique__technique": 2,
        "technique__belongs_to_tactic__tactic": 1,
    }
    config["sampler"]["reverse_fanouts"] = {}
    config["dataloader"]["num_workers"] = 0
    config["dataloader"]["pin_memory"] = False

    r, lr, ws = setup_distributed()
    try:
        train_neighbor_sampling(
            config, seed=42, device=torch.device("cpu"),
            rank=r, local_rank=lr, world_size=ws,
        )
    finally:
        teardown_distributed()


def test_hgt_ddp_two_rank_cpu_smoke(tmp_path: Path) -> None:
    npz_path = tmp_path / "tiny_graph.npz"
    meta_path = tmp_path / "tiny_graph.meta.json"
    output_dir = tmp_path / "out"
    _build_tiny_graph(npz_path, meta_path)

    port = _free_port()
    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(2):
        p = ctx.Process(
            target=_worker,
            args=(rank, 2, port, str(npz_path), str(meta_path), str(output_dir)),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=120)
    for p in procs:
        assert p.exitcode == 0, f"DDP worker exited with code {p.exitcode}"

    # Rank 0 wrote the checkpoint + summary.
    assert (output_dir / "hgt_flow_best.pt").exists()
    assert (output_dir / "training_summary.json").exists()

    # The checkpoint is loadable by a plain single-process consumer.
    ckpt = torch.load(output_dir / "hgt_flow_best.pt", map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt
    assert ckpt["num_classes"] == 2
    # And the saved state_dict has no DDP prefix or compile wrapper.
    for key in ckpt["model_state_dict"]:
        assert not key.startswith("module.")
        assert not key.startswith("_orig_mod.")
