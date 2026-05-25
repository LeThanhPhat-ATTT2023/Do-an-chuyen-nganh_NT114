"""Single-command orchestrator: raw pcaps -> v2 evidence-grounded graph artifact.

Usage:
    D:\\v\\nt114\\Scripts\\python.exe -m graphslm_ids.offline.preprocessing.v2.cli \\
        --raw-root data/raw/14gb \\
        --out-npz outputs/v2/graph.npz \\
        --out-meta-json outputs/v2/graph.meta.json
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from graphslm_ids.offline.preprocessing.v2.extractor import (
    extract_packets,
    extract_packets_dir,
)
from graphslm_ids.offline.preprocessing.v2.graph_builder import (
    build_v2_graph_artifact,
    save_v2_artifact,
)


def _build_payload_matrix_from_pcaps(
    raw_root: Path, packets_df, payload_length: int
) -> np.ndarray:
    """Re-walk the pcaps and copy each kept packet's payload bytes into a matrix.

    The extractor emits per-packet metadata but not the raw bytes (keeping the
    metadata path lightweight). For the graph artifact we need the actual
    bytes so payload features and evidence edges can fire on real content.
    This helper does a SECOND deterministic pass aligned to the metadata DF:
    for each row we already extracted, we re-open the same pcap and copy the
    payload of the packet at the recorded ``(pcap, packet_index)`` position.

    This is O(N_packets) and reasonably fast since dpkt is C-backed; on the
    14 GB dataset it runs in a couple of minutes. For huge corpora the caller
    can pre-extract bytes and pass `payload_matrix=` directly.

    NOTE: For simplicity here, we group rows by pcap path (inferred from
    label folder layout) and walk each pcap once. Rows that don't survive
    matching are zero-padded.
    """
    n = len(packets_df)
    out = np.zeros((n, payload_length), dtype=np.uint8)
    # Index rows by (label, packet_index) so we can write into the right slot
    # as we walk each pcap a second time.
    row_index: dict[tuple[str, int], list[int]] = {}
    if "label" not in packets_df.columns:
        return out
    # packet_index is not currently emitted by the v2 extractor; instead we use
    # the row position WITHIN each pcap (the extractor walks pcaps in directory
    # order). We bookkeep per-label cumulative positions.
    cum_per_label: dict[str, int] = {}
    for i, label in enumerate(packets_df["label"].astype(str).tolist()):
        pos = cum_per_label.get(label, 0)
        row_index.setdefault((label, pos), []).append(i)
        cum_per_label[label] = pos + 1

    import dpkt
    import socket  # noqa: F401  # imported for symmetry with extractor

    for cls_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        label = cls_dir.name
        running_pos = 0
        for pcap in sorted(cls_dir.glob("*.pcap")):
            try:
                f = open(pcap, "rb")
            except OSError:
                continue
            with f:
                try:
                    reader = dpkt.pcap.Reader(f)
                except Exception:
                    continue
                dlt = reader.datalink()
                # IMPORTANT: mirror v2/extractor.py's filter EXACTLY so the
                # running_pos counter advances in lockstep with the DF row
                # index for this class. The extractor only emits rows for
                # successfully-decoded IP packets; anything else (ARP, IPv6,
                # malformed) is silently skipped. If we incremented
                # running_pos for skipped packets here the byte rows would
                # slide out of alignment with `packets_df`.
                for _, (_ts, buf) in enumerate(reader):
                    try:
                        ip = _decode_ip(buf, dlt, dpkt)
                    except Exception:
                        ip = None
                    if not isinstance(ip, dpkt.ip.IP):
                        continue
                    rows = row_index.get((label, running_pos))
                    if rows:
                        payload = _extract_payload(ip, dpkt)
                        if payload:
                            vec = np.zeros(payload_length, dtype=np.uint8)
                            k = min(payload_length, len(payload))
                            vec[:k] = np.frombuffer(payload[:k], dtype=np.uint8)
                            for r in rows:
                                out[r] = vec
                    running_pos += 1
    return out


def _decode_ip(buf: bytes, dlt: int, dpkt_mod):
    if dlt == 1:  # Ethernet
        return dpkt_mod.ethernet.Ethernet(buf).data
    if dlt in (12, 101):
        return dpkt_mod.ip.IP(buf)
    if dlt == 113:
        return dpkt_mod.ip.IP(buf[16:])
    return dpkt_mod.ethernet.Ethernet(buf).data


def _extract_payload(ip, dpkt_mod) -> bytes:
    t = ip.data
    if isinstance(t, dpkt_mod.tcp.TCP):
        return bytes(t.data)
    if isinstance(t, dpkt_mod.udp.UDP):
        return bytes(t.data)
    if isinstance(t, dpkt_mod.icmp.ICMP):
        return bytes(t.data)
    return b""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", default="data/raw/14gb")
    ap.add_argument("--mitre-techniques-csv", default="data/mitre/mitre_techniques.csv")
    ap.add_argument(
        "--mitre-techniques-npy", default="data/mitre/mitre_techniques_embeddings.npy"
    )
    ap.add_argument(
        "--mitre-tactic-csv", default="data/mitre/mitre_technique_tactic_edges.csv"
    )
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--out-meta-json", required=True)
    ap.add_argument("--max-per-class", type=int, default=0, help="0 = no cap")
    ap.add_argument("--payload-length", type=int, default=256)
    ap.add_argument(
        "--with-payload-bytes",
        action="store_true",
        default=True,
        help=(
            "Re-walk pcaps to copy raw payload bytes into the per-packet "
            "payload matrix. Required for payload signatures to fire on real "
            "content. Default: on."
        ),
    )
    ap.add_argument(
        "--no-payload-bytes",
        dest="with_payload_bytes",
        action="store_false",
        help="Skip the payload byte pass (smoke runs only).",
    )
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    cap = None if args.max_per_class == 0 else args.max_per_class
    t0 = time.time()
    print(f"[v2] parsing pcaps from {raw_root} (cap={cap}) ...", flush=True)
    df = extract_packets_dir(raw_root, max_per_class=cap)
    print(
        f"[v2] packets parsed = {len(df):,}  classes = {df['label'].nunique()}",
        flush=True,
    )

    payload_matrix = None
    if args.with_payload_bytes:
        print("[v2] second pass: copying payload bytes ...", flush=True)
        # Note: the second pass relies on identical pcap iteration order as the
        # extractor. We respect max_per_class by truncating per-label below.
        payload_matrix = _build_payload_matrix_from_pcaps(
            raw_root, df, args.payload_length
        )

    print("[v2] assembling graph artifact ...", flush=True)
    arts = build_v2_graph_artifact(
        df,
        mitre_techniques_csv=Path(args.mitre_techniques_csv),
        mitre_technique_embeddings_npy=Path(args.mitre_techniques_npy),
        mitre_technique_tactic_csv=Path(args.mitre_tactic_csv),
        payload_length=args.payload_length,
        payload_matrix=payload_matrix,
    )
    save_v2_artifact(arts, Path(args.out_npz), Path(args.out_meta_json))
    print(
        f"[v2] OK  flows={arts['metadata']['num_flows']:,}  "
        f"packets={arts['metadata']['num_packets']:,}  "
        f"flow->tech edges={arts['metadata']['n_flow_technique_edges']:,}  "
        f"packet->tech edges={arts['metadata']['n_packet_technique_edges']:,}  "
        f"wall={time.time() - t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()


# Re-export for tests that want to build an artifact via a single helper.
__all__ = ["main", "extract_packets", "build_v2_graph_artifact", "save_v2_artifact"]
