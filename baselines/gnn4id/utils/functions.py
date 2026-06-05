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
_READ_CHUNK_ROWS = 20_000  # CSV read chunk for row-counting + subsampled streaming

# Columns excluded from flow features (metadata / packet-level list strings).
# nfstream may write custom plugin fields as "pkt_hex" or "udps.pkt_hex" depending on version.
_EXCLUDE = frozenset([
    "label", "src_ip", "dst_ip", "src_mac", "dst_mac", "src_oui", "dst_oui", "id",
    "application_name", "application_category_name", "requested_server_name",
    "client_fingerprint", "server_fingerprint", "user_agent", "content_type",
    # Unix timestamps in milliseconds (~1.7e12) — catastrophic for un-normalized GNNs
    "bidirectional_first_seen_ms", "bidirectional_last_seen_ms",
    "src2dst_first_seen_ms", "src2dst_last_seen_ms",
    "dst2src_first_seen_ms", "dst2src_last_seen_ms",
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
        # Store raw packet bytes as uint8 (values are 0-255) — 4x smaller on disk
        # and in RAM than float32. The model casts uint8 → float at forward time.
        g["packet"].x = torch.from_numpy(np.asarray(pkt_feat, dtype=np.uint8))  # (P, 1500) uint8

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
            arr = np.zeros(_PACKET_BYTES, dtype=np.uint8)
            n = min(len(raw), _PACKET_BYTES)
            if n:
                arr[:n] = np.frombuffer(raw[:n], dtype=np.uint8)
            feats.append(arr)
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


def _count_csv_rows(csv_path: str) -> int:
    """Count data rows (excluding header) without loading the whole file into RAM."""
    n = 0
    for chunk in pd.read_csv(csv_path, usecols=[0], chunksize=_READ_CHUNK_ROWS):
        n += len(chunk)
    return n


def stream_graphs_from_csv(
    csv_path: str,
    label_idx: int,
    max_flows: int | None = None,
    seed: int = 42,
) -> Iterator[HeteroData]:
    """Yield one HeteroData per (optionally subsampled) flow row in ``csv_path``.

    * ``max_flows is None`` → build every row (legacy behaviour; loads the CSV).
    * ``max_flows`` set → build a uniform random subset of that many rows, read in
      chunks (two-pass) so peak RAM stays bounded by one chunk + the kept graphs,
      never the whole multi-GB CSV.
    """
    if max_flows is None:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            g = NIDSDataset._build_graph(row, label_idx)
            if g is not None:
                yield g
        return

    n_total = _count_csv_rows(csv_path)
    if n_total <= max_flows:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            g = NIDSDataset._build_graph(row, label_idx)
            if g is not None:
                yield g
        return

    rng = np.random.default_rng(seed)
    selected = {int(i) for i in rng.choice(n_total, size=max_flows, replace=False)}
    offset = 0
    for chunk in pd.read_csv(csv_path, chunksize=_READ_CHUNK_ROWS):
        for i, (_, row) in enumerate(chunk.iterrows()):
            if (offset + i) in selected:
                g = NIDSDataset._build_graph(row, label_idx)
                if g is not None:
                    yield g
        offset += len(chunk)


class ShardWriter:
    """Streams per-CSV graphs to disk shards, persisting an incremental manifest.

    Designed for crash/disk-full safety: the manifest is rewritten atomically after
    every CSV, so a re-run resumes cleanly — already-built CSV stems are skipped and
    a per-class flow cap is honoured across the whole dataset (even multiple PCAPs
    per class). Pair with delete-CSV-after-build to bound disk usage.
    """

    def __init__(
        self,
        manifest_path: str,
        label_mapping: dict[str, int],
        shard_dir: str | None = None,
        max_graphs_per_shard: int = 50_000,
        max_flows_per_class: int | None = None,
        seed: int = 42,
    ):
        self.manifest_p = Path(manifest_path)
        self.label_mapping = label_mapping
        if shard_dir is None:
            self.shard_dir_p = self.manifest_p.parent / (self.manifest_p.stem + "_shards")
        else:
            self.shard_dir_p = Path(shard_dir)
        self.shard_dir_p.mkdir(parents=True, exist_ok=True)
        self.max_graphs_per_shard = max_graphs_per_shard
        self.max_flows_per_class = max_flows_per_class
        self.seed = seed

        self.shards: list[str] = []
        self.total = 0
        self.built_stems: set[str] = set()
        self.per_class: dict[str, int] = {}
        if self.manifest_p.exists():            # resume from a prior (possibly partial) run
            try:
                m = json.loads(self.manifest_p.read_text())
            except Exception:
                m = {}
            if m.get("format") == MANIFEST_FORMAT:
                self.shards = list(m.get("shards", []))
                self.total = int(m.get("num_graphs", 0))
                self.built_stems = set(m.get("built_stems", []))
                self.per_class = dict(m.get("per_class", {}))

    @staticmethod
    def stem_for(csv_path: str, label: str) -> str:
        return f"{label}__{Path(csv_path).stem}"

    def already_built(self, csv_path: str, label: str) -> bool:
        return self.stem_for(csv_path, label) in self.built_stems

    def add_csv(self, csv_path: str, label: str) -> int:
        """Build + flush shards for one CSV. Returns #graphs added (0 if skipped)."""
        stem = self.stem_for(csv_path, label)
        if stem in self.built_stems:
            _LOG.info("  skip (already built): %s", stem)
            return 0
        if label not in self.label_mapping:
            _LOG.warning("  label %r not in label_mapping, skipping %s", label, csv_path)
            return 0
        label_idx = self.label_mapping[label]

        max_flows = None
        if self.max_flows_per_class is not None:
            remaining = self.max_flows_per_class - self.per_class.get(label, 0)
            if remaining <= 0:
                _LOG.info("  class %s at cap (%d) — skip %s",
                          label, self.max_flows_per_class, stem)
                self.built_stems.add(stem)
                self._persist()
                return 0
            max_flows = remaining

        buf: list[HeteroData] = []
        seq = 0
        added = 0
        for g in stream_graphs_from_csv(csv_path, label_idx, max_flows=max_flows, seed=self.seed):
            buf.append(g)
            if len(buf) >= self.max_graphs_per_shard:
                added += self._flush(buf, stem, seq)
                seq += 1
                buf = []          # release references so GC can reclaim the shard
        added += self._flush(buf, stem, seq)

        self.built_stems.add(stem)
        self.per_class[label] = self.per_class.get(label, 0) + added
        self._persist()
        return added

    def _flush(self, buf: list[HeteroData], stem: str, seq: int) -> int:
        if not buf:
            return 0
        shard_path = self.shard_dir_p / f"{stem}.{seq:03d}.pt"
        torch.save(buf, str(shard_path))
        try:
            rel = shard_path.relative_to(self.manifest_p.parent)
        except ValueError:
            rel = shard_path
        self.shards.append(str(rel).replace("\\", "/"))
        self.total += len(buf)
        _LOG.info("  shard %s (%d graphs, total %d)", shard_path.name, len(buf), self.total)
        return len(buf)

    def _manifest_dict(self) -> dict:
        return {
            "format": MANIFEST_FORMAT,
            "label_mapping": self.label_mapping,
            "num_graphs": self.total,
            "shards": self.shards,
            "built_stems": sorted(self.built_stems),
            "per_class": self.per_class,
        }

    def _persist(self) -> None:
        self.manifest_p.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_p.with_suffix(self.manifest_p.suffix + ".tmp")
        tmp.write_text(json.dumps(self._manifest_dict(), indent=2))
        tmp.replace(self.manifest_p)   # atomic — never leave a half-written manifest

    def finalize(self) -> dict:
        self._persist()
        return self._manifest_dict()


def write_graph_shards(
    csv_files: list[tuple[str, str]],
    label_mapping: dict[str, int],
    manifest_path: str,
    shard_dir: str | None = None,
    max_graphs_per_shard: int = 50_000,
    max_flows_per_class: int | None = None,
    seed: int = 42,
) -> dict:
    """Stream every CSV's graphs to disk shards and write a manifest JSON.

    Thin wrapper over :class:`ShardWriter` for the non-interleaved path and tests.
    Peak RAM is bounded by ``max_graphs_per_shard`` (plus one read chunk); the
    final artifact layout is::

        <manifest_path>                          # JSON manifest (+ resume state)
        <shard_dir>/<label>__<csv_stem>.NNN.pt    # torch.save(list[HeteroData])
    """
    w = ShardWriter(manifest_path, label_mapping, shard_dir,
                    max_graphs_per_shard, max_flows_per_class, seed)
    for csv_path, label in csv_files:
        w.add_csv(csv_path, label)
    return w.finalize()


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
