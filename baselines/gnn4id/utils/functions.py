"""PyG NIDSDataset: builds HeteroData graphs from nfstream CSVs (dynamic label_mapping).

Two build paths share one set of (static) graph builders:
  * ``NIDSDataset``         — legacy in-RAM path: builds every graph into a list.
  * ``stream_graphs_from_csv`` / ``write_graph_shards`` — streaming path that
    builds graphs per-CSV and flushes them to disk shards, so peak RAM is bounded
    by one shard instead of the whole dataset (avoids the build-step OOM).
"""
from __future__ import annotations
import ast
import json
import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset, HeteroData

_LOG = logging.getLogger("gnn4id.functions")
MANIFEST_FORMAT = "gnn4id-shards-v1"

# Columns excluded from flow features (metadata / packet-level list strings).
# nfstream may write custom plugin fields as "pkt_hex" or "udps.pkt_hex" depending on version.
_EXCLUDE = frozenset([
    "label", "src_ip", "dst_ip", "src_mac", "dst_mac", "src_oui", "dst_oui", "id",
    "application_name", "application_category_name", "requested_server_name",
    "client_fingerprint", "server_fingerprint", "user_agent", "content_type",
    # bare names (older nfstream)
    "pkt_hex", "pkt_delta", "pkt_dir",
    "pkt_ip_size", "pkt_transport_size", "pkt_payload_size",
    "syn", "cwr", "ece", "urg", "ack", "psh", "rst", "fin",
    # src_dst helpers added by additional_features (should be dropped, but guard here)
    "src_dst_ip", "src_dst_encoded", "dst_ip_encoded",
])

_PACKET_BYTES = 1500  # GNN4ID default payload width


def _row_get(row: pd.Series, name: str, default="[]") -> object:
    """Return row[name], trying 'udps.<name>' prefix as fallback (nfstream >=6.4 style)."""
    if name in row.index:
        return row[name]
    udps_name = f"udps.{name}"
    if udps_name in row.index:
        return row[udps_name]
    return default


def _safe_parse_list(val) -> list | None:
    try:
        result = ast.literal_eval(str(val))
        return result if isinstance(result, list) else [result]
    except Exception:
        return None


class NIDSDataset(Dataset):
    """Heterogeneous graph dataset for GNN4ID (13-class adapted).

    Args:
        csv_files:     List of (csv_path, label_str) tuples.
        label_mapping: Dict mapping label_str → integer index.
    """

    def __init__(self, csv_files: list[tuple[str, str]], label_mapping: dict[str, int]):
        super().__init__()
        self.label_mapping = label_mapping
        self._graphs: list[HeteroData] = []

        for csv_path, label in csv_files:
            label_idx = label_mapping[label]
            self._graphs.extend(stream_graphs_from_csv(csv_path, label_idx))

    # ── PyG Dataset interface ──────────────────────────────────────────────────

    def len(self) -> int:
        return len(self._graphs)

    def get(self, idx: int) -> HeteroData:
        return self._graphs[idx]

    # ── Internal builders (static: shared by the streaming path) ────────────────

    @staticmethod
    def _build_graph(row: pd.Series, label_idx: int) -> HeteroData | None:
        flow_feat = NIDSDataset._flow_features(row)
        pkt_feat, pkt_delta = NIDSDataset._packet_features(row)
        if pkt_feat is None or len(pkt_feat) == 0:
            return None
        n_pkts = len(pkt_feat)

        g = HeteroData()
        g["flow"].x = torch.tensor([flow_feat], dtype=torch.float)  # (1, F)
        g["flow"].y = torch.tensor([label_idx], dtype=torch.long)
        g["packet"].x = torch.tensor(np.array(pkt_feat, dtype=np.float32))  # (P, 1500)

        # contains: flow(0) → each packet
        g["flow", "contains", "packet"].edge_index = torch.stack([
            torch.zeros(n_pkts, dtype=torch.long),
            torch.arange(n_pkts, dtype=torch.long),
        ])
        g["flow", "contains", "packet"].edge_attr = torch.tensor(
            NIDSDataset._contain_edge_attr(row, n_pkts), dtype=torch.float
        )  # (P, 4)

        # link: packet_i → packet_{i+1}
        if n_pkts > 1:
            g["packet", "next_packet", "packet"].edge_index = torch.stack([
                torch.arange(n_pkts - 1, dtype=torch.long),
                torch.arange(1, n_pkts, dtype=torch.long),
            ])
            g["packet", "next_packet", "packet"].edge_attr = torch.tensor(
                [[float(d)] for d in pkt_delta[1:]], dtype=torch.float
            )  # (P-1, 1)
        else:
            g["packet", "next_packet", "packet"].edge_index = torch.zeros((2, 0), dtype=torch.long)
            g["packet", "next_packet", "packet"].edge_attr = torch.zeros((0, 1), dtype=torch.float)

        return g

    @staticmethod
    def _flow_features(row: pd.Series) -> list[float]:
        feats = []
        for col in row.index:
            # Skip metadata, packet-level list strings, and all udps.* columns
            if col in _EXCLUDE or col.startswith("udps."):
                continue
            try:
                feats.append(float(row[col]))
            except (ValueError, TypeError):
                feats.append(0.0)
        return feats

    @staticmethod
    def _packet_features(row: pd.Series) -> tuple[list[list[float]] | None, list[float] | None]:
        hexes = _safe_parse_list(_row_get(row, "pkt_hex"))
        deltas = _safe_parse_list(_row_get(row, "pkt_delta"))
        if hexes is None or deltas is None:
            return None, None
        feats = []
        for h in hexes:
            raw = bytes.fromhex(h) if isinstance(h, str) and h else b""
            arr = np.zeros(_PACKET_BYTES, dtype=np.float32)
            n = min(len(raw), _PACKET_BYTES)
            if n:
                arr[:n] = np.frombuffer(raw[:n], dtype=np.uint8)
            feats.append(arr.tolist())
        deltas_f = [float(d) for d in deltas]
        return feats, deltas_f

    @staticmethod
    def _contain_edge_attr(row: pd.Series, n_pkts: int) -> list[list[float]]:
        dirs = _safe_parse_list(_row_get(row, "pkt_dir")) or []
        ips  = _safe_parse_list(_row_get(row, "pkt_ip_size")) or []
        trs  = _safe_parse_list(_row_get(row, "pkt_transport_size")) or []
        pls  = _safe_parse_list(_row_get(row, "pkt_payload_size")) or []
        attrs = []
        for i in range(n_pkts):
            attrs.append([
                float(dirs[i]) if i < len(dirs) else 0.0,
                float(ips[i])  if i < len(ips)  else 0.0,
                float(trs[i])  if i < len(trs)  else 0.0,
                float(pls[i])  if i < len(pls)  else 0.0,
            ])
        return attrs


# ── Streaming build + on-disk shards (bounded build-step RAM) ────────────────


def stream_graphs_from_csv(csv_path: str, label_idx: int) -> Iterator[HeteroData]:
    """Yield one HeteroData per valid flow row in ``csv_path`` without buffering.

    This is the streaming counterpart to ``NIDSDataset.__init__``: it never holds
    more than a single row's graph at a time, so the caller controls peak memory.
    """
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        g = NIDSDataset._build_graph(row, label_idx)
        if g is not None:
            yield g


def write_graph_shards(
    csv_files: list[tuple[str, str]],
    label_mapping: dict[str, int],
    manifest_path: str,
    shard_dir: str | None = None,
    max_graphs_per_shard: int = 50_000,
) -> dict:
    """Stream every CSV's graphs to disk shards and write a manifest JSON.

    Peak RAM is bounded by ``max_graphs_per_shard`` (plus one CSV's DataFrame),
    not by the whole dataset. Returns the manifest dict (also written to disk).

    Layout::

        <manifest_path>                         # JSON manifest
        <shard_dir>/<label>__<csv_stem>.NNN.pt   # torch.save(list[HeteroData])
    """
    manifest_p = Path(manifest_path)
    if shard_dir is None:
        shard_dir_p = manifest_p.parent / (manifest_p.stem + "_shards")
    else:
        shard_dir_p = Path(shard_dir)
    shard_dir_p.mkdir(parents=True, exist_ok=True)

    shards: list[str] = []
    total = 0

    def _flush(buf: list[HeteroData], stem: str, seq: int) -> None:
        nonlocal total
        if not buf:
            return
        shard_path = shard_dir_p / f"{stem}.{seq:03d}.pt"
        torch.save(buf, str(shard_path))
        # store path relative to the manifest so the artifact is relocatable
        try:
            rel = shard_path.relative_to(manifest_p.parent)
        except ValueError:
            rel = shard_path
        shards.append(str(rel).replace("\\", "/"))
        total += len(buf)
        _LOG.info("  shard %s (%d graphs, running total %d)", shard_path.name, len(buf), total)

    for csv_path, label in csv_files:
        if label not in label_mapping:
            _LOG.warning("  label %r not in label_mapping, skipping %s", label, csv_path)
            continue
        label_idx = label_mapping[label]
        stem = f"{label}__{Path(csv_path).stem}"
        buf: list[HeteroData] = []
        seq = 0
        for g in stream_graphs_from_csv(csv_path, label_idx):
            buf.append(g)
            if len(buf) >= max_graphs_per_shard:
                _flush(buf, stem, seq)
                seq += 1
                buf = []          # release references so GC can reclaim the shard
        _flush(buf, stem, seq)

    manifest = {
        "format": MANIFEST_FORMAT,
        "label_mapping": label_mapping,
        "num_graphs": total,
        "shards": shards,
    }
    manifest_p.parent.mkdir(parents=True, exist_ok=True)
    manifest_p.write_text(json.dumps(manifest, indent=2))
    return manifest


def load_graphs(path: str) -> tuple[list[HeteroData], dict[str, int]]:
    """Load graphs + label_mapping from either a shard manifest or a legacy .pt.

    Accepts:
      * a ``*.manifest.json`` (or a directory containing one) → reads & concatenates shards;
      * a legacy single ``graphs.pt`` (``{"graphs": [...], "label_mapping": {...}}``).

    The streaming build keeps the BUILD step's RAM bounded; this loader still
    concatenates shards into one list for training (training-time RAM unchanged).
    """
    p = Path(path)
    if p.is_dir():
        manifests = sorted(p.glob("*.manifest.json"))
        if not manifests:
            raise FileNotFoundError(f"No *.manifest.json found in directory {p}")
        p = manifests[0]

    if p.suffix == ".json":
        manifest = json.loads(p.read_text())
        if manifest.get("format") != MANIFEST_FORMAT:
            raise ValueError(f"Unrecognized manifest format: {manifest.get('format')!r}")
        label_mapping = manifest["label_mapping"]
        graphs: list[HeteroData] = []
        for rel in manifest["shards"]:
            shard_path = (p.parent / rel)
            graphs.extend(torch.load(str(shard_path), map_location="cpu", weights_only=False))
        return graphs, label_mapping

    # Legacy single-file artifact
    saved = torch.load(str(p), map_location="cpu", weights_only=False)
    return saved["graphs"], saved["label_mapping"]
