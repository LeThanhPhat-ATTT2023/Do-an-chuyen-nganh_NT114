"""Assemble the v2 tri-modal heterogeneous graph artifact.

Inputs: a packet dataframe (from the v2 extractor) + MITRE artefacts on disk.
Outputs: a dictionary of numpy arrays + a metadata dict matching the v1 schema
shape (so the HGT loader can read both v1 and v2 with a tiny version switch),
plus the artifact_version='v2' marker used to gate the new loader path.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from graphslm_ids.offline.preprocessing.v2.evidence_edges import (
    build_evidence_edges,
)
from graphslm_ids.offline.preprocessing.v2.flows import (
    assign_flows,
    build_flow_features,
)
from graphslm_ids.offline.preprocessing.v2.payload_features import (
    FEATURE_DIM as PAYLOAD_FEATURE_DIM,
    FEATURE_NAMES as PAYLOAD_FEATURE_NAMES,
    aggregate_flow_payload_features,
    compute_packet_payload_features,
)


def _build_payload_matrix(df: pd.DataFrame, payload_length: int) -> np.ndarray:
    """Approximate a per-packet payload byte matrix when raw bytes are unavailable.

    The v2 extractor records ``payload_len`` but not the bytes themselves (the
    raw-byte pipeline is built lazily by the CLI). For unit tests and smoke
    runs we fabricate a deterministic matrix from the per-packet metadata so
    every downstream step still has a stable, well-formed input.

    Producers that already have payload bytes should pass them in directly via
    a future CLI hook; for now the byte matrix is a zero-padded placeholder
    sized correctly. The hashed-ngram features therefore depend on whatever
    bytes the caller injects; tests use real bytes via the demo pcap fixture
    re-parsing path.
    """
    n = len(df)
    if n == 0:
        return np.zeros((0, payload_length), dtype=np.uint8)
    out = np.zeros((n, payload_length), dtype=np.uint8)
    # The extractor doesn't store byte content; for the smoke artifact we keep
    # the matrix zeroed. Real builds will plug in the payload-256 array.
    return out


def _encode_categorical(values: pd.Series) -> tuple[np.ndarray, dict[str, int]]:
    """Sorted-string -> int encoding (matches v1 ``_encode_series`` semantics)."""
    uniq = sorted({str(v) for v in values.tolist()})
    mapping = {v: i for i, v in enumerate(uniq)}
    encoded = np.asarray([mapping[str(v)] for v in values.tolist()], dtype=np.int64)
    return encoded, mapping


def _load_mitre_tactics(
    technique_tactic_csv: Path,
    techniques_csv: Path,
    technique_id_to_idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, int], int]:
    """Read MITRE technique<->tactic edges and produce (edge_index, edge_attr, mapping, n_tactics).

    Mirrors `_build_technique_tactic_arrays` from the v1 builder but trimmed for v2.
    """
    edges_df = pd.read_csv(technique_tactic_csv)
    required = {"technique_id", "tactic_shortname"}
    if not required.issubset(edges_df.columns):
        raise ValueError(
            f"technique_tactic_csv must contain {sorted(required)}, got {list(edges_df.columns)}"
        )
    tactics = sorted({str(t) for t in edges_df["tactic_shortname"].tolist()})
    tactic_to_idx = {t: i for i, t in enumerate(tactics)}

    src: list[int] = []
    dst: list[int] = []
    for _, row in edges_df.iterrows():
        tid = str(row["technique_id"])
        tac = str(row["tactic_shortname"])
        if tid not in technique_id_to_idx:
            continue
        src.append(technique_id_to_idx[tid])
        dst.append(tactic_to_idx[tac])
    if src:
        edge_index = np.vstack(
            [np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64)]
        )
        edge_attr = np.ones((edge_index.shape[1], 1), dtype=np.float32)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, 1), dtype=np.float32)
    return edge_index, edge_attr, tactic_to_idx, len(tactics)


def build_v2_graph_artifact(
    packets_df: pd.DataFrame,
    mitre_techniques_csv: Path,
    mitre_technique_embeddings_npy: Path,
    mitre_technique_tactic_csv: Path,
    payload_length: int = 256,
    payload_matrix: np.ndarray | None = None,
) -> dict[str, Any]:
    """Assemble the v2 graph artifact from a packets dataframe.

    Args:
      packets_df: output of :func:`v2.extractor.extract_packets_dir` (or
        :func:`extract_packets`). Must contain the v2 column schema.
      mitre_techniques_csv: path to ``data/mitre/mitre_techniques.csv``.
      mitre_technique_embeddings_npy: path to ``mitre_techniques_embeddings.npy``.
      mitre_technique_tactic_csv: path to ``mitre_technique_tactic_edges.csv``.
      payload_length: per-packet byte window (defaults to 256, matches v1).
      payload_matrix: optional ``(N_packets, payload_length)`` uint8 matrix. If
        omitted, a zero matrix is used (payload features collapse to byte=0
        signatures; suitable for smoke runs only).

    Returns a dict with all node/edge arrays + a ``metadata`` dict and
    ``artifact_version="v2"``.
    """
    if packets_df.empty:
        raise ValueError("Cannot build a v2 artifact from an empty packet frame")

    # 1. Flow assembly + per-flow features.
    tagged = assign_flows(packets_df)
    feats, feats_meta = build_flow_features(tagged)
    # Preserve packet ordering for the payload + contain edges below.
    tagged = tagged.reset_index(drop=True)

    # 2. flow_x, flow_y from features.
    flow_feature_columns = [c for c in feats.columns if c != "label"]
    flow_x = feats[flow_feature_columns].to_numpy(dtype=np.float32)
    flow_y_enc, label_mapping = _encode_categorical(feats["label"])

    # 3. Packet payload matrix + per-flow packet indices.
    if payload_matrix is None:
        payload_matrix = _build_payload_matrix(tagged, payload_length)
    elif payload_matrix.shape[0] != len(tagged):
        raise ValueError(
            f"payload_matrix has {payload_matrix.shape[0]} rows but the tagged "
            f"packet frame has {len(tagged)} rows"
        )
    packet_payload_idx: dict[str, np.ndarray] = {}
    for fid, idx in tagged.groupby("flow_id").groups.items():
        packet_payload_idx[fid] = np.asarray(list(idx), dtype=np.int64)
    # Ensure ordering matches the feats index (i.e. flow_x rows).
    packet_payload_idx_ordered = {
        fid: packet_payload_idx.get(fid, np.array([], dtype=np.int64))
        for fid in feats.index.tolist()
    }
    packet_x, packet_feature_names = aggregate_flow_payload_features(
        payload_matrix, packet_payload_idx_ordered
    )
    # The v1 schema kept packet features at packet-node granularity. v2 keeps a
    # PARALLEL representation at packet-node granularity too — one row per
    # packet in `tagged` — by computing features directly from payload_matrix.
    if len(tagged) > 0:
        packet_x_per_packet = np.stack(
            [
                compute_packet_payload_features(payload_matrix[i], payload_length)
                for i in range(len(tagged))
            ],
            axis=0,
        )
    else:
        packet_x_per_packet = np.zeros((0, PAYLOAD_FEATURE_DIM), dtype=np.float32)

    # 4. MITRE technique features + technique_id_to_idx mapping.
    techniques_df = pd.read_csv(mitre_techniques_csv)
    if "technique_id" not in techniques_df.columns:
        raise ValueError("mitre_techniques_csv must contain a 'technique_id' column")
    technique_x = np.load(mitre_technique_embeddings_npy)
    if technique_x.shape[0] != len(techniques_df):
        raise ValueError(
            "mitre_techniques_embeddings.npy row count does not match mitre_techniques.csv"
        )
    technique_id_to_idx = {
        tid: i for i, tid in enumerate(techniques_df["technique_id"].astype(str).tolist())
    }

    # 5. Technique <-> tactic edges + tactic ids.
    tt_edge_index, tt_edge_attr, tactic_to_idx, n_tactics = _load_mitre_tactics(
        mitre_technique_tactic_csv, mitre_techniques_csv, technique_id_to_idx
    )
    tactic_x = np.arange(n_tactics, dtype=np.int64)[:, None]

    # 6. contain edges (flow -> packet) and next_packet edges.
    flow_id_to_idx = {fid: i for i, fid in enumerate(feats.index.tolist())}
    contain_src = np.asarray(
        [flow_id_to_idx[fid] for fid in tagged["flow_id"].tolist()], dtype=np.int64
    )
    contain_dst = np.arange(len(tagged), dtype=np.int64)
    contain_edge_index = np.vstack([contain_src, contain_dst])

    # next_packet edges: consecutive packets within the same flow (already
    # sorted by ts inside assign_flows).
    if len(tagged) > 0:
        same_flow = (
            tagged["flow_id"].shift(1) == tagged["flow_id"]
        ).to_numpy().copy()
        same_flow[0] = False
        link_dst = np.where(same_flow)[0]
        link_src = link_dst - 1
        prev_ts = tagged["ts"].to_numpy()
        delta_t = (prev_ts[link_dst] - prev_ts[link_src]).astype(np.float32)
        link_edge_index = np.vstack([link_src.astype(np.int64), link_dst.astype(np.int64)])
        link_edge_attr = delta_t[:, None]
    else:
        link_edge_index = np.empty((2, 0), dtype=np.int64)
        link_edge_attr = np.empty((0, 1), dtype=np.float32)

    # 7. Evidence-grounded technique edges.
    ev = build_evidence_edges(
        feats[flow_feature_columns],
        payload_matrix,
        packet_payload_idx_ordered,
        technique_id_to_idx,
    )

    artifact: dict[str, Any] = {
        "flow_x": flow_x,
        "flow_y": flow_y_enc.astype(np.int64),
        "packet_x": packet_x_per_packet,  # per-packet payload features
        "packet_x_flow_mean": packet_x,    # per-flow mean (auxiliary, useful for ablations)
        "technique_x": technique_x.astype(np.float32),
        "tactic_x": tactic_x,
        "contain_edge_index": contain_edge_index,
        "link_edge_index": link_edge_index,
        "link_edge_attr": link_edge_attr,
        "flow_technique_edge_index": ev["flow_technique_edge_index"],
        "flow_technique_edge_attr": ev["flow_technique_edge_attr"],
        "packet_technique_edge_index": ev["packet_technique_edge_index"],
        "packet_technique_edge_attr": ev["packet_technique_edge_attr"],
        "technique_tactic_edge_index": tt_edge_index,
        "technique_tactic_edge_attr": tt_edge_attr,
    }
    artifact["metadata"] = {
        "artifact_version": "v2",
        "num_flows": int(flow_x.shape[0]),
        "num_packets": int(packet_x_per_packet.shape[0]),
        "num_techniques": int(technique_x.shape[0]),
        "num_tactics": int(n_tactics),
        "payload_length": int(payload_length),
        "flow_feature_names": flow_feature_columns,
        "packet_feature_names": PAYLOAD_FEATURE_NAMES,
        "label_mapping": label_mapping,
        "tactic_mapping": tactic_to_idx,
        "n_flow_technique_edges": int(ev["flow_technique_edge_index"].shape[1]),
        "n_packet_technique_edges": int(ev["packet_technique_edge_index"].shape[1]),
        "n_technique_tactic_edges": int(tt_edge_index.shape[1]),
        "n_contain_edges": int(contain_edge_index.shape[1]),
        "n_link_edges": int(link_edge_index.shape[1]),
        "proto_counts": feats_meta.get("proto_counts", {}),
    }
    return artifact


def save_v2_artifact(arts: dict[str, Any], out_npz: Path, out_meta_json: Path) -> None:
    """Persist the artifact to disk in the same layout v1 used (NPZ + meta.json)."""
    out_npz = Path(out_npz)
    out_meta_json = Path(out_meta_json)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_meta_json.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in arts.items() if isinstance(v, np.ndarray)}
    np.savez(str(out_npz), **arrays)
    with open(out_meta_json, "w", encoding="utf-8") as f:
        json.dump(_jsonable(arts["metadata"]), f, indent=2)


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy / pandas types so json.dump won't choke."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
