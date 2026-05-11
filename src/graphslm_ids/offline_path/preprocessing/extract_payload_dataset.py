from __future__ import annotations

import argparse
import glob
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graphslm_ids.offline_path.preprocessing.pcap_payload_extractor import build_payload_dataset
from graphslm_ids.offline_path.preprocessing.graph_csv_builder import build_graph_csv_tables
from graphslm_ids.utils.io import ensure_dir, write_json


def resolve_input_pcaps(patterns: list[str]) -> list[Path]:
    """Resolve one or more glob patterns into sorted, unique PCAP file paths."""
    discovered: dict[str, Path] = {}

    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        for match in matches:
            path = Path(match)
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".pcap", ".pcapng"}:
                continue
            discovered[str(path.resolve())] = path

    return sorted(discovered.values(), key=lambda p: str(p))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract fixed-length payload vectors (default 256 bytes) from PCAP files."
    )
    parser.add_argument(
        "--input-glob",
        nargs="+",
        required=True,
        help="One or more glob patterns that match .pcap/.pcapng files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/interim/payload_dataset",
        help="Directory where payload_256.npy, metadata.csv, and stats.json are written.",
    )
    parser.add_argument(
        "--payload-length",
        type=int,
        default=256,
        help="Fixed payload length after truncate+pad.",
    )
    parser.add_argument(
        "--max-packets-per-file",
        type=int,
        default=None,
        help="Optional cap on packets read from each PCAP file.",
    )
    parser.add_argument(
        "--include-empty-payload",
        action="store_true",
        help="Include packets that have empty payloads (all-zero vectors after padding).",
    )
    parser.add_argument(
        "--log-every-packets",
        type=int,
        default=0,
        help="Print progress every N packets per PCAP file (0 to disable).",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle output rows before saving.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when --shuffle is set.")
    parser.add_argument(
        "--save-npz",
        action="store_true",
        help="Also save compressed payload vectors as payload_256.npz.",
    )
    parser.add_argument(
        "--export-graph-csv",
        action="store_true",
        help="Export flow_nodes.csv, packet_nodes.csv, contain_edges.csv, and link_edges.csv.",
    )
    parser.add_argument(
        "--graph-flow-timeout-seconds",
        type=float,
        default=30.0,
        help="Idle time threshold (seconds) for rolling over to a new flow when exporting graph CSV.",
    )
    parser.add_argument(
        "--graph-max-packets-per-flow",
        type=int,
        default=20,
        help="Maximum packet count per flow segment when exporting graph CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pcap_paths = resolve_input_pcaps(args.input_glob)
    if not pcap_paths:
        raise SystemExit("No PCAP files found. Check --input-glob patterns.")

    log_every = int(args.log_every_packets)
    if log_every <= 0:
        log_every = None

    payload_matrix, metadata = build_payload_dataset(
        pcap_paths=pcap_paths,
        payload_length=args.payload_length,
        max_packets_per_file=args.max_packets_per_file,
        include_empty_payload=args.include_empty_payload,
        log_every=log_every,
    )

    if args.shuffle and payload_matrix.shape[0] > 0:
        rng = np.random.default_rng(args.seed)
        indices = rng.permutation(payload_matrix.shape[0])
        payload_matrix = payload_matrix[indices]
        metadata = metadata.iloc[indices].reset_index(drop=True)

    output_dir = ensure_dir(Path(args.output_dir))
    payload_path = output_dir / "payload_256.npy"
    metadata_path = output_dir / "metadata.csv"
    stats_path = output_dir / "stats.json"

    np.save(payload_path, payload_matrix)
    if args.save_npz:
        np.savez_compressed(output_dir / "payload_256.npz", payload_matrix)

    metadata.to_csv(metadata_path, index=False)

    graph_output_files: dict[str, str] = {}
    graph_counts: dict[str, int] = {}
    if args.export_graph_csv:
        graph_tables = build_graph_csv_tables(
            metadata,
            flow_timeout_seconds=args.graph_flow_timeout_seconds,
            max_packets_per_flow=args.graph_max_packets_per_flow,
        )
        flow_nodes_path = output_dir / "flow_nodes.csv"
        packet_nodes_path = output_dir / "packet_nodes.csv"
        contain_edges_path = output_dir / "contain_edges.csv"
        link_edges_path = output_dir / "link_edges.csv"

        graph_tables.flow_nodes.to_csv(flow_nodes_path, index=False)
        graph_tables.packet_nodes.to_csv(packet_nodes_path, index=False)
        graph_tables.contain_edges.to_csv(contain_edges_path, index=False)
        graph_tables.link_edges.to_csv(link_edges_path, index=False)

        graph_output_files = {
            "flow_nodes_csv": str(flow_nodes_path),
            "packet_nodes_csv": str(packet_nodes_path),
            "contain_edges_csv": str(contain_edges_path),
            "link_edges_csv": str(link_edges_path),
        }
        graph_counts = {
            "flow_nodes": int(graph_tables.flow_nodes.shape[0]),
            "packet_nodes": int(graph_tables.packet_nodes.shape[0]),
            "contain_edges": int(graph_tables.contain_edges.shape[0]),
            "link_edges": int(graph_tables.link_edges.shape[0]),
        }

    label_counts: dict[str, int] = {}
    protocol_counts: dict[str, int] = {}
    if not metadata.empty:
        label_counts = {str(k): int(v) for k, v in metadata["label"].value_counts().items()}
        protocol_counts = {str(k): int(v) for k, v in metadata["protocol"].value_counts().items()}

    stats = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_pcap_files": len(pcap_paths),
        "num_packets": int(payload_matrix.shape[0]),
        "payload_length": int(args.payload_length),
        "label_counts": label_counts,
        "protocol_counts": protocol_counts,
        "output_files": {
            "payload_npy": str(payload_path),
            "metadata_csv": str(metadata_path),
        },
        "graph_csv": {
            "enabled": bool(args.export_graph_csv),
            "flow_timeout_seconds": float(args.graph_flow_timeout_seconds),
            "max_packets_per_flow": int(args.graph_max_packets_per_flow),
            "output_files": graph_output_files,
            "counts": graph_counts,
        },
    }
    write_json(stats_path, stats)

    print(f"[OK] PCAP files: {len(pcap_paths)}")
    print(f"[OK] Packets extracted: {payload_matrix.shape[0]}")
    print(f"[OK] Payload matrix saved: {payload_path}")
    print(f"[OK] Metadata saved: {metadata_path}")
    if args.export_graph_csv:
        print("[OK] Graph CSV exported:")
        print(f"     - flow_nodes: {graph_output_files['flow_nodes_csv']}")
        print(f"     - packet_nodes: {graph_output_files['packet_nodes_csv']}")
        print(f"     - contain_edges: {graph_output_files['contain_edges_csv']}")
        print(f"     - link_edges: {graph_output_files['link_edges_csv']}")
    print(f"[OK] Stats saved: {stats_path}")


if __name__ == "__main__":
    main()
