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
