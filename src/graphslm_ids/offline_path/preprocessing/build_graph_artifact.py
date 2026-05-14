from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from graphslm_ids.offline_path.preprocessing.graph_artifact_builder import build_graph_artifact
from graphslm_ids.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build graph artifact directly from metadata.csv and payload_256.npy."
    )
    parser.add_argument("--metadata-csv", required=True, help="Path to metadata.csv")
    parser.add_argument("--payload-npy", required=True, help="Path to payload_256.npy")
    parser.add_argument(
        "--output-npz",
        default="data/processed/graph_artifact.npz",
        help="Output NPZ path for graph arrays.",
    )
    parser.add_argument(
        "--output-meta-json",
        default=None,
        help="Optional output JSON path for graph metadata. Default: <output-npz>.meta.json",
    )
    parser.add_argument("--flow-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-packets-per-flow", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    metadata_csv = Path(args.metadata_csv)
    payload_npy = Path(args.payload_npy)
    output_npz = Path(args.output_npz)
    output_meta_json = (
        Path(args.output_meta_json)
        if args.output_meta_json is not None
        else output_npz.with_suffix(".meta.json")
    )

    metadata = pd.read_csv(metadata_csv)
    payload_matrix = np.load(payload_npy)

    artifact = build_graph_artifact(
        metadata=metadata,
        payload_matrix=payload_matrix,
        flow_timeout_seconds=args.flow_timeout_seconds,
        max_packets_per_flow=args.max_packets_per_flow,
    )

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **artifact.arrays)

    export_meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_csv": str(metadata_csv),
        "payload_npy": str(payload_npy),
        "graph_output_npz": str(output_npz),
        "flow_timeout_seconds": float(args.flow_timeout_seconds),
        "max_packets_per_flow": int(args.max_packets_per_flow),
    }
    export_meta.update(artifact.metadata)
    write_json(output_meta_json, export_meta)

    print(f"[OK] Graph artifact: {output_npz}")
    print(f"[OK] Graph metadata: {output_meta_json}")
    print(
        "[OK] Counts -> "
        f"flows={artifact.metadata['num_flows']}, "
        f"packets={artifact.metadata['num_packets']}, "
        f"contain_edges={artifact.metadata['num_contain_edges']}, "
        f"link_edges={artifact.metadata['num_link_edges']}"
    )


if __name__ == "__main__":
    main()
