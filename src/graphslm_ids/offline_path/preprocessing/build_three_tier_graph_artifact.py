from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graphslm_ids.offline_path.preprocessing.graph_artifact_builder import build_graph_artifact
from graphslm_ids.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build three-tier heterogeneous graph artifact with flow-packet structure and "
            "packet/flow to MITRE technique tactical edges from cosine similarity."
        )
    )
    parser.add_argument("--metadata-csv", required=True, help="Path to payload metadata.csv")
    parser.add_argument("--payload-npy", required=True, help="Path to payload_256.npy")
    parser.add_argument(
        "--student-embedding-npy",
        required=True,
        help="Path to student embedding matrix (rows aligned with payload rows).",
    )
    parser.add_argument(
        "--mitre-techniques-csv",
        required=True,
        help="Path to MITRE techniques CSV containing technique_id.",
    )
    parser.add_argument(
        "--mitre-technique-embeddings-npy",
        required=True,
        help="Path to MITRE technique embedding matrix aligned with MITRE techniques CSV rows.",
    )
    parser.add_argument(
        "--mitre-technique-tactic-edges-csv",
        required=True,
        help="Path to CSV edges technique_id -> tactic_shortname.",
    )
    parser.add_argument(
        "--output-npz",
        default="data/processed/graph_artifact_3tier_t082_k5.npz",
        help="Output NPZ path for three-tier graph arrays.",
    )
    parser.add_argument(
        "--output-meta-json",
        default=None,
        help="Optional metadata JSON path. Default: <output-npz>.meta.json",
    )
    parser.add_argument("--flow-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-packets-per-flow", type=int, default=20)
    parser.add_argument("--similarity-threshold", type=float, default=0.82)
    parser.add_argument("--packet-top-k", type=int, default=5)
    parser.add_argument("--flow-top-k", type=int, default=5)
    return parser.parse_args()


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def _select_top_k_above_threshold(
    similarity_matrix: np.ndarray,
    top_k: int,
    threshold: float,
) -> list[tuple[int, int, float]]:
    num_rows, num_cols = similarity_matrix.shape
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    edges: list[tuple[int, int, float]] = []
    for src_idx in range(num_rows):
        row = similarity_matrix[src_idx]
        k = min(top_k, num_cols)
        if k <= 0:
            continue

        candidate_indices = np.argpartition(-row, kth=k - 1)[:k]
        candidate_pairs = sorted(
            ((int(dst_idx), float(row[int(dst_idx)])) for dst_idx in candidate_indices),
            key=lambda x: x[1],
            reverse=True,
        )
        for dst_idx, score in candidate_pairs:
            if score >= threshold:
                edges.append((src_idx, dst_idx, score))
    return edges


def _build_technique_tactic_arrays(
    techniques_df: pd.DataFrame,
    technique_tactic_edges_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[str, int]]:
    technique_id_to_idx = {
        str(technique_id): idx
        for idx, technique_id in enumerate(techniques_df["technique_id"].astype(str).tolist())
    }

    tactic_shortnames = sorted({str(x) for x in technique_tactic_edges_df["tactic_shortname"].astype(str).tolist()})
    tactic_to_idx = {name: idx for idx, name in enumerate(tactic_shortnames)}

    src_list: list[int] = []
    dst_list: list[int] = []
    for row in technique_tactic_edges_df.itertuples(index=False):
        technique_id = str(row.technique_id)
        tactic_shortname = str(row.tactic_shortname)
        if technique_id not in technique_id_to_idx:
            continue
        if tactic_shortname not in tactic_to_idx:
            continue
        src_list.append(technique_id_to_idx[technique_id])
        dst_list.append(tactic_to_idx[tactic_shortname])

    if src_list:
        edge_index = np.vstack(
            [
                np.asarray(src_list, dtype=np.int64),
                np.asarray(dst_list, dtype=np.int64),
            ]
        )
        edge_attr = np.ones((len(src_list), 1), dtype=np.float32)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, 1), dtype=np.float32)

    return edge_index, edge_attr, technique_id_to_idx, tactic_to_idx


def main() -> None:
    args = parse_args()

    metadata_csv = Path(args.metadata_csv)
    payload_npy = Path(args.payload_npy)
    student_embedding_npy = Path(args.student_embedding_npy)
    mitre_techniques_csv = Path(args.mitre_techniques_csv)
    mitre_technique_embeddings_npy = Path(args.mitre_technique_embeddings_npy)
    mitre_technique_tactic_edges_csv = Path(args.mitre_technique_tactic_edges_csv)
    output_npz = Path(args.output_npz)
    output_meta_json = (
        Path(args.output_meta_json)
        if args.output_meta_json is not None
        else output_npz.with_suffix(".meta.json")
    )

    metadata = pd.read_csv(metadata_csv)
    payload_matrix = np.load(payload_npy)

    base_artifact = build_graph_artifact(
        metadata=metadata,
        payload_matrix=payload_matrix,
        flow_timeout_seconds=args.flow_timeout_seconds,
        max_packets_per_flow=args.max_packets_per_flow,
    )

    student_embedding = np.asarray(np.load(student_embedding_npy), dtype=np.float32)
    if student_embedding.ndim != 2:
        raise ValueError("student embedding matrix must be 2D.")
    if student_embedding.shape[0] != payload_matrix.shape[0]:
        raise ValueError("student embedding rows must align with payload rows.")

    techniques_df = pd.read_csv(mitre_techniques_csv)
    if "technique_id" not in techniques_df.columns:
        raise ValueError("MITRE techniques CSV must contain technique_id column.")

    mitre_embedding = np.asarray(np.load(mitre_technique_embeddings_npy), dtype=np.float32)
    if mitre_embedding.ndim != 2:
        raise ValueError("MITRE technique embedding matrix must be 2D.")
    if mitre_embedding.shape[0] != techniques_df.shape[0]:
        raise ValueError("MITRE embedding rows must align with MITRE techniques CSV rows.")
    if mitre_embedding.shape[1] != student_embedding.shape[1]:
        raise ValueError("Embedding dimensions must match between student and MITRE techniques.")

    packet_payload_row_index = base_artifact.arrays["packet_payload_row_index"]
    packet_embedding = student_embedding[packet_payload_row_index]

    packet_embedding_norm = _normalize_rows(packet_embedding)
    mitre_embedding_norm = _normalize_rows(mitre_embedding)

    packet_technique_similarity = packet_embedding_norm @ mitre_embedding_norm.T
    packet_technique_edges = _select_top_k_above_threshold(
        packet_technique_similarity,
        top_k=args.packet_top_k,
        threshold=args.similarity_threshold,
    )

    if packet_technique_edges:
        packet_technique_edge_index = np.vstack(
            [
                np.asarray([e[0] for e in packet_technique_edges], dtype=np.int64),
                np.asarray([e[1] for e in packet_technique_edges], dtype=np.int64),
            ]
        )
        packet_technique_edge_attr = np.asarray(
            [[e[2]] for e in packet_technique_edges], dtype=np.float32
        )
    else:
        packet_technique_edge_index = np.empty((2, 0), dtype=np.int64)
        packet_technique_edge_attr = np.empty((0, 1), dtype=np.float32)

    contain_edge_index = base_artifact.arrays["contain_edge_index"]
    num_flows = int(base_artifact.metadata["num_flows"])
    num_techniques = int(techniques_df.shape[0])
    flow_technique_sum = np.zeros((num_flows, num_techniques), dtype=np.float32)
    flow_technique_count = np.zeros((num_flows, 1), dtype=np.float32)

    if contain_edge_index.shape[1] > 0:
        flow_indices = contain_edge_index[0].astype(np.int64)
        packet_indices = contain_edge_index[1].astype(np.int64)
        np.add.at(flow_technique_sum, flow_indices, packet_technique_similarity[packet_indices])
        np.add.at(flow_technique_count, flow_indices, 1.0)

    flow_technique_count = np.maximum(flow_technique_count, 1.0)
    flow_technique_similarity = flow_technique_sum / flow_technique_count

    flow_technique_edges = _select_top_k_above_threshold(
        flow_technique_similarity,
        top_k=args.flow_top_k,
        threshold=args.similarity_threshold,
    )

    if flow_technique_edges:
        flow_technique_edge_index = np.vstack(
            [
                np.asarray([e[0] for e in flow_technique_edges], dtype=np.int64),
                np.asarray([e[1] for e in flow_technique_edges], dtype=np.int64),
            ]
        )
        flow_technique_edge_attr = np.asarray(
            [[e[2]] for e in flow_technique_edges], dtype=np.float32
        )
    else:
        flow_technique_edge_index = np.empty((2, 0), dtype=np.int64)
        flow_technique_edge_attr = np.empty((0, 1), dtype=np.float32)

    technique_tactic_edges_df = pd.read_csv(mitre_technique_tactic_edges_csv)
    required_edge_cols = {"technique_id", "tactic_shortname"}
    if not required_edge_cols.issubset(technique_tactic_edges_df.columns):
        raise ValueError(
            f"MITRE technique-tactic edges CSV must contain columns {sorted(required_edge_cols)}."
        )

    (
        technique_tactic_edge_index,
        technique_tactic_edge_attr,
        technique_id_to_idx,
        tactic_shortname_to_idx,
    ) = _build_technique_tactic_arrays(techniques_df, technique_tactic_edges_df)

    arrays = dict(base_artifact.arrays)
    arrays.update(
        {
            "technique_x": mitre_embedding,
            "packet_semantic_x": packet_embedding,
            "packet_technique_edge_index": packet_technique_edge_index,
            "packet_technique_edge_attr": packet_technique_edge_attr,
            "flow_technique_edge_index": flow_technique_edge_index,
            "flow_technique_edge_attr": flow_technique_edge_attr,
            "technique_tactic_edge_index": technique_tactic_edge_index,
            "technique_tactic_edge_attr": technique_tactic_edge_attr,
        }
    )

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)

    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_csv": str(metadata_csv),
        "payload_npy": str(payload_npy),
        "student_embedding_npy": str(student_embedding_npy),
        "mitre_techniques_csv": str(mitre_techniques_csv),
        "mitre_technique_embeddings_npy": str(mitre_technique_embeddings_npy),
        "mitre_technique_tactic_edges_csv": str(mitre_technique_tactic_edges_csv),
        "graph_output_npz": str(output_npz),
        "flow_timeout_seconds": float(args.flow_timeout_seconds),
        "max_packets_per_flow": int(args.max_packets_per_flow),
        "similarity_threshold": float(args.similarity_threshold),
        "packet_top_k": int(args.packet_top_k),
        "flow_top_k": int(args.flow_top_k),
        "num_flows": int(base_artifact.metadata["num_flows"]),
        "num_packets": int(base_artifact.metadata["num_packets"]),
        "num_techniques": int(mitre_embedding.shape[0]),
        "num_tactics": int(len(tactic_shortname_to_idx)),
        "num_contain_edges": int(base_artifact.metadata["num_contain_edges"]),
        "num_link_edges": int(base_artifact.metadata["num_link_edges"]),
        "num_packet_technique_edges": int(packet_technique_edge_index.shape[1]),
        "num_flow_technique_edges": int(flow_technique_edge_index.shape[1]),
        "num_technique_tactic_edges": int(technique_tactic_edge_index.shape[1]),
        "payload_length": int(base_artifact.metadata["payload_length"]),
        "embedding_dim": int(student_embedding.shape[1]),
        "label_mapping": base_artifact.metadata.get("label_mapping", {}),
        "protocol_mapping": base_artifact.metadata.get("protocol_mapping", {}),
        "flow_feature_names": base_artifact.metadata.get("flow_feature_names", []),
        "technique_id_to_idx": technique_id_to_idx,
        "tactic_shortname_to_idx": tactic_shortname_to_idx,
    }
    write_json(output_meta_json, meta)

    print(f"[OK] Three-tier graph artifact: {output_npz}")
    print(f"[OK] Three-tier metadata: {output_meta_json}")
    print(
        "[OK] Counts -> "
        f"flows={meta['num_flows']}, "
        f"packets={meta['num_packets']}, "
        f"techniques={meta['num_techniques']}, "
        f"tactics={meta['num_tactics']}, "
        f"packet_technique_edges={meta['num_packet_technique_edges']}, "
        f"flow_technique_edges={meta['num_flow_technique_edges']}, "
        f"technique_tactic_edges={meta['num_technique_tactic_edges']}"
    )


if __name__ == "__main__":
    main()
