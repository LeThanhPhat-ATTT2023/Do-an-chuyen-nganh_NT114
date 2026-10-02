"""Tiny synthetic v3 artifact shared by the federated tests (CPU, well under 1 s)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

LABELS = {"Benign": 0, "XSS": 1, "DDoS-ICMP_Flood": 2}
FLOW_FEATURES = ["proto", "src_port", "dst_port", "sig_a", "sig_b", "sig_c"]
TECHNIQUES = {"T1059": 0, "T1190": 1, "T1498": 2, "T1595": 3}


def host_features_reference(
    src_host: np.ndarray, dst_host: np.ndarray, dst_port: np.ndarray, num_hosts: int
) -> np.ndarray:
    """Host tier features over ALL flows, by the same rule as graph_builder._build_host_tier:
    out-degree, in-degree, #distinct dst ports and #distinct dst hosts per source host."""
    out = np.zeros((num_hosts, 4), dtype=np.float32)
    ports: list[set[int]] = [set() for _ in range(num_hosts)]
    peers: list[set[int]] = [set() for _ in range(num_hosts)]
    for s, d, p in zip(src_host.tolist(), dst_host.tolist(), dst_port.tolist(), strict=True):
        out[s, 0] += 1.0
        out[d, 1] += 1.0
        ports[s].add(int(p))
        peers[s].add(d)
    out[:, 2] = [len(s) for s in ports]
    out[:, 3] = [len(s) for s in peers]
    return out


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
    from_host = flows % 3
    to_host = 3 + (flows % 3)
    host_x = host_features_reference(from_host, to_host, flow_x[:, 2].astype(np.int64), 6)
    arrays = {
        "flow_x": flow_x,
        "flow_y": y,
        "packet_x": packet_x,
        "host_x": host_x,
        "technique_x": rng.normal(size=(len(TECHNIQUES), 5)).astype(np.float32),
        "contain_edge_index": np.stack([np.repeat(flows, 2), np.arange(num_packets)]),
        "link_edge_index": np.stack([2 * flows, 2 * flows + 1]),
        "link_edge_attr": np.full((num_flows, 1), 0.01, dtype=np.float32),
        "from_host_edge_index": np.stack([flows, from_host]),
        "from_host_edge_attr": np.ones((num_flows, 1), dtype=np.float32),
        "to_host_edge_index": np.stack([flows, to_host]),
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
