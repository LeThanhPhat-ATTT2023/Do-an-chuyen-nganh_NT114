"""PyG NIDSDataset: builds HeteroData graphs from nfstream CSVs (dynamic label_mapping)."""
from __future__ import annotations
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset, HeteroData

# Columns that carry packet-level list strings or metadata — excluded from flow features.
_EXCLUDE = frozenset([
    "label", "src_ip", "dst_ip", "src_mac", "dst_mac", "id",
    "pkt_hex", "pkt_delta", "pkt_dir",
    "pkt_ip_size", "pkt_transport_size", "pkt_payload_size",
    "syn", "cwr", "ece", "urg", "ack", "psh", "rst", "fin",
])

_PACKET_BYTES = 1500  # GNN4ID default payload width


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
            df = pd.read_csv(csv_path)
            label_idx = label_mapping[label]
            for _, row in df.iterrows():
                g = self._build_graph(row, label_idx)
                if g is not None:
                    self._graphs.append(g)

    # ── PyG Dataset interface ──────────────────────────────────────────────────

    def len(self) -> int:
        return len(self._graphs)

    def get(self, idx: int) -> HeteroData:
        return self._graphs[idx]

    # ── Internal builders ──────────────────────────────────────────────────────

    def _build_graph(self, row: pd.Series, label_idx: int) -> HeteroData | None:
        flow_feat = self._flow_features(row)
        pkt_feat, pkt_delta = self._packet_features(row)
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
            self._contain_edge_attr(row, n_pkts), dtype=torch.float
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

    def _flow_features(self, row: pd.Series) -> list[float]:
        feats = []
        for col in row.index:
            if col in _EXCLUDE:
                continue
            try:
                feats.append(float(row[col]))
            except (ValueError, TypeError):
                feats.append(0.0)
        return feats

    def _packet_features(self, row: pd.Series) -> tuple[list[list[float]] | None, list[float] | None]:
        hexes = _safe_parse_list(row.get("pkt_hex", "[]"))
        deltas = _safe_parse_list(row.get("pkt_delta", "[]"))
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

    def _contain_edge_attr(self, row: pd.Series, n_pkts: int) -> list[list[float]]:
        dirs = _safe_parse_list(row.get("pkt_dir", "[]")) or []
        ips = _safe_parse_list(row.get("pkt_ip_size", "[]")) or []
        trs = _safe_parse_list(row.get("pkt_transport_size", "[]")) or []
        pls = _safe_parse_list(row.get("pkt_payload_size", "[]")) or []
        attrs = []
        for i in range(n_pkts):
            attrs.append([
                float(dirs[i]) if i < len(dirs) else 0.0,
                float(ips[i])  if i < len(ips)  else 0.0,
                float(trs[i])  if i < len(trs)  else 0.0,
                float(pls[i])  if i < len(pls)  else 0.0,
            ])
        return attrs
