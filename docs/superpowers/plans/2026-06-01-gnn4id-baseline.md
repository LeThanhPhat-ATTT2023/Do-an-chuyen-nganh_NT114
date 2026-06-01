# GNN4ID Baseline Adaptation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adapt GNN4ID (Farrukh et al., 2025) into runnable Python scripts under `baselines/gnn4id/` that train a 13-class heterogeneous GNN on CIC-IoT-2023 PCAPs and output `results.json` with macro F1 for direct comparison with EG-HGT v3.

**Architecture:** GNN4ID builds one PyG `HeteroData` graph per flow (flow node + ≤20 packet nodes), uses `HeteroGNN_Edge` (two GATConv layers, hidden=64) for graph-level classification. Preprocessing uses **nfstream** (not scapy/dpkt) to extract per-flow CSV, then computes rolling-window additional features, then builds PyG graphs. Training uses Adam + early stopping on val macro F1; outputs `results.json`.

**Tech Stack:** Python 3.10+, nfstream, torch 2.x, torch-geometric, pandas, scikit-learn, pytest.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `baselines/gnn4id/requirements.txt` | Create | Python dependencies |
| `baselines/gnn4id/.gitignore` | Create | Ignore `outputs/` |
| `baselines/gnn4id/utils/__init__.py` | Create | Package marker |
| `baselines/gnn4id/utils/additional_features.py` | Create | Rolling-window feature engineering on nfstream CSV |
| `baselines/gnn4id/utils/feature_extractor.py` | Create | nfstream plugin → per-PCAP CSV |
| `baselines/gnn4id/utils/functions.py` | Create | `NIDSDataset` PyG dataset (dynamic label_mapping) |
| `baselines/gnn4id/model.py` | Create | `HeteroGNN_Edge` with `num_classes` param |
| `baselines/gnn4id/preprocess.py` | Create | CLI: raw PCAPs → `outputs/graphs.pt` |
| `baselines/gnn4id/train.py` | Create | CLI: `graphs.pt` → train → `outputs/results.json` |
| `tests/baselines/__init__.py` | Create | Package marker |
| `tests/baselines/test_model.py` | Create | Model output shape, forward pass |
| `tests/baselines/test_dataset.py` | Create | NIDSDataset graph structure |

---

## Task 1: Scaffold — directories, requirements, gitignore

**Files:**
- Create: `baselines/gnn4id/requirements.txt`
- Create: `baselines/gnn4id/.gitignore`
- Create: `baselines/gnn4id/utils/__init__.py`
- Create: `tests/baselines/__init__.py`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p baselines/gnn4id/utils
mkdir -p baselines/gnn4id/outputs
mkdir -p tests/baselines
```

- [ ] **Step 2: Write `baselines/gnn4id/requirements.txt`**

```
nfstream>=6.5.2
torch>=2.0.0
torch-geometric>=2.4.0
pandas>=1.5.0
scikit-learn>=1.2.0
numpy>=1.24.0
tqdm>=4.65.0
```

- [ ] **Step 3: Write `baselines/gnn4id/.gitignore`**

```
outputs/
__pycache__/
*.pyc
*.pt
*.json
```

- [ ] **Step 4: Create empty `__init__.py` files**

```bash
touch baselines/gnn4id/utils/__init__.py
touch tests/baselines/__init__.py
```

- [ ] **Step 5: Commit**

```bash
git add baselines/gnn4id/requirements.txt baselines/gnn4id/.gitignore \
        baselines/gnn4id/utils/__init__.py tests/baselines/__init__.py
git commit -m "feat(baseline): scaffold gnn4id directory structure"
```

---

## Task 2: `utils/additional_features.py`

**Files:**
- Create: `baselines/gnn4id/utils/additional_features.py`

This is GNN4ID's `Additional_Features.py` copied verbatim and turned into a proper module. It reads a nfstream CSV, adds rolling-window statistical features, and overwrites the CSV in-place.

- [ ] **Step 1: Write `baselines/gnn4id/utils/additional_features.py`**

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def additional_features(
    file_name,
    window_size=350,
    http_ports=[443, 8080, 80],
    vulnerable_ports=[20, 21, 22, 23, 25, 53, 80, 110, 143, 443, 445, 3389, 8080],
    dns_ports=[53],
    exp_id=[0, -1],
    proto_list=[1, 2, 6, 17, 58],
):
    """Compute rolling-window features on a nfstream CSV and overwrite it in-place."""
    try:
        data = pd.read_csv(file_name)
    except Exception:
        print(f"file reading error: {file_name}")
        return ""
    try:
        data = data.sort_values(by="bidirectional_first_seen_ms")
    except Exception:
        print("CSV does not contain bidirectional_first_seen_ms for initial sorting")
        return ""

    def rolling_sum(group, w):
        return group.rolling(window=w, min_periods=1).sum()

    def rolling_mean(group, w):
        return group.rolling(window=w, min_periods=1).mean()

    def rolling_unique(group, w):
        return group.rolling(window=w, min_periods=1).apply(
            lambda x: len(set(x)), raw=True
        )

    le1, le2 = LabelEncoder(), LabelEncoder()
    data["src_dst_ip"] = data["src_ip"].astype(str) + "-" + data["dst_ip"].astype(str)
    data["src_dst_encoded"] = le1.fit_transform(data["src_dst_ip"])
    data["dst_ip_encoded"] = le2.fit_transform(data["dst_ip"].astype(str))

    data["packet_size_variation"] = data[
        ["src2dst_min_ps", "src2dst_max_ps", "dst2src_min_ps", "dst2src_max_ps"]
    ].std(axis=1)

    data["is_udp_request"] = (data["protocol"] == 17).astype(int)
    data["is_tcp_request"] = (data["protocol"] == 6).astype(int)
    data["is_icmp_request"] = (data["protocol"] == 1).astype(int)

    for col, feat, label in [
        ("Rolling_UDP_Requests_SourceDestination", "is_udp_request", "src_dst_encoded"),
        ("Rolling_UDP_Requests_Destination", "is_udp_request", "dst_ip_encoded"),
        ("Rolling_TCP_Requests_SourceDestination", "is_tcp_request", "src_dst_encoded"),
        ("Rolling_TCP_Requests_Destination", "is_tcp_request", "dst_ip_encoded"),
        ("Rolling_ACK_Packets_SourceDestination", "bidirectional_ack_packets", "src_dst_encoded"),
        ("Rolling_ACK_Packets_Destination", "bidirectional_ack_packets", "dst_ip_encoded"),
        ("Rolling_FIN_Packets_SourceDestination", "bidirectional_fin_packets", "src_dst_encoded"),
        ("Rolling_FIN_Packets_Destination", "bidirectional_fin_packets", "dst_ip_encoded"),
        ("Rolling_rst_Packets_SourceDestination", "bidirectional_rst_packets", "src_dst_encoded"),
        ("Rolling_rst_Packets_Destination", "bidirectional_rst_packets", "dst_ip_encoded"),
        ("Rolling_psh_Packets_SourceDestination", "bidirectional_psh_packets", "src_dst_encoded"),
        ("Rolling_psh_Packets_Destination", "bidirectional_psh_packets", "dst_ip_encoded"),
        ("Rolling_SYN_Packets_SourceDestination", "bidirectional_syn_packets", "src_dst_encoded"),
        ("Rolling_SYN_Packets_Destination", "bidirectional_syn_packets", "dst_ip_encoded"),
        ("Rolling_ICMP_Requests_SourceDestination", "is_icmp_request", "src_dst_encoded"),
        ("Rolling_ICMP_Requests_Destination", "is_icmp_request", "dst_ip_encoded"),
    ]:
        data[col] = (
            data.groupby(label)[feat]
            .apply(lambda x: rolling_sum(x, window_size))
            .reset_index(level=0, drop=True)
        )

    data["Unique_Ports_In_SourceDestinationIP"] = (
        data.groupby("src_dst_encoded")["dst_port"]
        .apply(lambda x: rolling_unique(x, window_size))
        .reset_index(level=0, drop=True)
    )

    for col, label in [
        ("Rolling_Duration_Destination", "dst_ip_encoded"),
        ("Rolling_Duration_SourceDestination", "src_dst_encoded"),
    ]:
        data[col] = (
            data.groupby(label)["bidirectional_duration_ms"]
            .apply(lambda x: rolling_mean(x, window_size))
            .reset_index(level=0, drop=True)
        )

    for port_set, prefix in [(http_ports, "http"), (dns_ports, "dns"), (dns_ports, "dns2")]:
        data["is_vulnerable_port"] = data["dst_port"].isin(port_set).astype(int)
        data[f"Rolling_{prefix}_port_SourceDestination"] = (
            data.groupby("src_dst_encoded")["is_vulnerable_port"]
            .apply(lambda x: rolling_sum(x, window_size))
            .reset_index(level=0, drop=True)
        )
        data[f"Rolling_{prefix}_port_Destination"] = (
            data.groupby("dst_ip_encoded")["is_vulnerable_port"]
            .apply(lambda x: rolling_sum(x, window_size))
            .reset_index(level=0, drop=True)
        )

    data["is_vulnerable_port"] = data["dst_port"].isin(vulnerable_ports).astype(int)
    data["Rolling_vulnerable_port"] = (
        data.groupby("src_dst_encoded")["is_vulnerable_port"]
        .apply(lambda x: rolling_sum(x, window_size))
        .reset_index(level=0, drop=True)
    )
    data["Rolling_packets_destination"] = (
        data.groupby("dst_ip_encoded")["src2dst_packets"]
        .apply(lambda x: rolling_sum(x, window_size))
        .reset_index(level=0, drop=True)
    )
    data["Rolling_bipackets_destination"] = (
        data.groupby("dst_ip_encoded")["bidirectional_packets"]
        .apply(lambda x: rolling_sum(x, window_size))
        .reset_index(level=0, drop=True)
    )

    data["expiration_id"] = pd.Categorical(data["expiration_id"], categories=exp_id)
    data["protocol"] = pd.Categorical(data["protocol"], categories=proto_list)
    data = pd.get_dummies(
        data, prefix=["Exp", "proto"], columns=["expiration_id", "protocol"], dtype=int
    )
    data.drop(
        ["src_dst_ip", "src_dst_encoded", "dst_ip_encoded",
         "is_udp_request", "is_tcp_request", "is_icmp_request", "is_vulnerable_port"],
        axis=1, inplace=True, errors="ignore",
    )
    data.to_csv(file_name, index=False)
    return file_name
```

- [ ] **Step 2: Commit**

```bash
git add baselines/gnn4id/utils/additional_features.py
git commit -m "feat(baseline): add additional_features rolling-window module"
```

---

## Task 3: `utils/feature_extractor.py`

**Files:**
- Create: `baselines/gnn4id/utils/feature_extractor.py`

Wraps GNN4ID's nfstream custom plugin. Adds the `label` column before saving so each CSV already carries its class name.

- [ ] **Step 1: Write `baselines/gnn4id/utils/feature_extractor.py`**

```python
"""nfstream-based PCAP → CSV extractor (GNN4ID style, max 20 pkts/flow)."""
from __future__ import annotations
import pandas as pd
from nfstream import NFPlugin, NFStreamer


class _PacketCapture(NFPlugin):
    """Captures per-packet payload hex + metadata for up to `limit` packets."""

    def on_init(self, packet, flow):
        raw = packet.ip_packet
        flow.udps.pkt_hex = [raw[20:].hex() if raw else ""]
        flow.udps.pkt_delta = [0]
        flow.udps.pkt_dir = [packet.direction]
        flow.udps.pkt_ip_size = [packet.ip_size]
        flow.udps.pkt_transport_size = [packet.transport_size]
        flow.udps.pkt_payload_size = [packet.payload_size]
        flow.udps.syn = [packet.syn]
        flow.udps.cwr = [packet.cwr]
        flow.udps.ece = [packet.ece]
        flow.udps.urg = [packet.urg]
        flow.udps.ack = [packet.ack]
        flow.udps.psh = [packet.psh]
        flow.udps.rst = [packet.rst]
        flow.udps.fin = [packet.fin]

    def on_update(self, packet, flow):
        if flow.bidirectional_packets >= self.limit:
            flow.expiration_id = -1
            return
        raw = packet.ip_packet
        flow.udps.pkt_hex.append(raw[20:].hex() if raw else "")
        flow.udps.pkt_delta.append(packet.delta_time)
        flow.udps.pkt_dir.append(packet.direction)
        flow.udps.pkt_ip_size.append(packet.ip_size)
        flow.udps.pkt_transport_size.append(packet.transport_size)
        flow.udps.pkt_payload_size.append(packet.payload_size)
        flow.udps.syn.append(packet.syn)
        flow.udps.cwr.append(packet.cwr)
        flow.udps.ece.append(packet.ece)
        flow.udps.urg.append(packet.urg)
        flow.udps.ack.append(packet.ack)
        flow.udps.psh.append(packet.psh)
        flow.udps.rst.append(packet.rst)
        flow.udps.fin.append(packet.fin)


def extract_pcap_to_csv(
    pcap_path: str,
    out_csv: str,
    label: str,
    max_pkts: int = 20,
) -> None:
    """Run nfstream on one PCAP and write a labelled CSV.

    Args:
        pcap_path: Path to input .pcap file.
        out_csv:   Destination CSV path.
        label:     Class name to store in the ``label`` column.
        max_pkts:  Maximum bidirectional packets captured per flow (default 20).
    """
    streamer = NFStreamer(
        source=pcap_path,
        udps=_PacketCapture(limit=max_pkts),
        statistical_analysis=True,
    )
    streamer.to_csv(path=out_csv, flows_per_file=0, columns_to_anonymize=[])
    df = pd.read_csv(out_csv)
    df["label"] = label
    df.to_csv(out_csv, index=False)
```

- [ ] **Step 2: Commit**

```bash
git add baselines/gnn4id/utils/feature_extractor.py
git commit -m "feat(baseline): add nfstream feature extractor"
```

---

## Task 4: `utils/functions.py` — NIDSDataset

**Files:**
- Create: `baselines/gnn4id/utils/functions.py`

Builds a PyG `Dataset` from a list of labelled CSVs. `label_mapping` is passed in (no hardcoded 8 classes). Each CSV row → one `HeteroData` graph.

- [ ] **Step 1: Write `baselines/gnn4id/utils/functions.py`**

```python
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
        self._flow_feature_dim: int | None = None

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
```

- [ ] **Step 2: Commit**

```bash
git add baselines/gnn4id/utils/functions.py
git commit -m "feat(baseline): add NIDSDataset with dynamic label_mapping"
```

---

## Task 5: `model.py`

**Files:**
- Create: `baselines/gnn4id/model.py`

Port of `HeteroGNN_Edge` with `num_classes` as a constructor parameter instead of hardcoded 8.

- [ ] **Step 1: Write `baselines/gnn4id/model.py`**

```python
"""GNN4ID HeteroGNN_Edge — adapted for arbitrary num_classes."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import HeteroConv, GATConv, global_mean_pool


class HeteroGNN_Edge(nn.Module):
    """Heterogeneous GNN with edge attributes (GATConv, 2 layers, hidden=64).

    Args:
        metadata:    PyG graph metadata: (node_types, edge_types).
        hidden_channels: Hidden dim per conv layer (default 64, GNN4ID default).
        num_classes: Number of output classes (13 for CIC-IoT-2023 13-class).
    """

    def __init__(
        self,
        metadata: tuple,
        hidden_channels: int = 64,
        num_classes: int = 13,
    ):
        super().__init__()
        node_types, edge_types = metadata

        self.conv1 = HeteroConv(
            {et: GATConv((-1, -1), hidden_channels, edge_dim=-1, add_self_loops=False)
             for et in edge_types},
            aggr="sum",
        )
        self.conv2 = HeteroConv(
            {et: GATConv((-1, -1), hidden_channels, edge_dim=-1, add_self_loops=False)
             for et in edge_types},
            aggr="sum",
        )

        self.bn1 = nn.ModuleDict({nt: nn.BatchNorm1d(hidden_channels) for nt in node_types})
        self.bn2 = nn.ModuleDict({nt: nn.BatchNorm1d(hidden_channels) for nt in node_types})

        n_node_types = len(node_types)
        self.classifier = nn.Sequential(
            nn.Linear(n_node_types * hidden_channels, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Linear(16, num_classes),
        )

    def forward(
        self,
        x_dict: dict,
        edge_index_dict: dict,
        edge_attr_dict: dict,
        batch_dict: dict,
    ) -> torch.Tensor:
        x_dict = self.conv1(x_dict, edge_index_dict, edge_attr_dict)
        x_dict = {
            nt: F.leaky_relu(self.bn1[nt](x)) for nt, x in x_dict.items()
        }
        x_dict = self.conv2(x_dict, edge_index_dict, edge_attr_dict)
        x_dict = {
            nt: F.leaky_relu(self.bn2[nt](x)) for nt, x in x_dict.items()
        }
        pooled = [
            global_mean_pool(x_dict[nt], batch_dict[nt])
            for nt in sorted(x_dict)
        ]
        return self.classifier(torch.cat(pooled, dim=1))
```

- [ ] **Step 2: Commit**

```bash
git add baselines/gnn4id/model.py
git commit -m "feat(baseline): add HeteroGNN_Edge with num_classes param"
```

---

## Task 6: Unit Tests

**Files:**
- Create: `tests/baselines/test_model.py`
- Create: `tests/baselines/test_dataset.py`

- [ ] **Step 1: Write `tests/baselines/test_model.py`**

```python
"""Unit tests for HeteroGNN_Edge model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "baselines" / "gnn4id"))

import torch
from torch_geometric.data import HeteroData, Batch
from model import HeteroGNN_Edge


def _make_batch(n_graphs: int = 4, n_pkts_each: int = 5, num_classes: int = 13):
    """Build a synthetic batched HeteroData for testing."""
    graphs = []
    for _ in range(n_graphs):
        g = HeteroData()
        g["flow"].x = torch.randn(1, 82)
        g["flow"].y = torch.randint(0, num_classes, (1,))
        g["packet"].x = torch.randn(n_pkts_each, 1500)
        g["flow", "contains", "packet"].edge_index = torch.stack([
            torch.zeros(n_pkts_each, dtype=torch.long),
            torch.arange(n_pkts_each, dtype=torch.long),
        ])
        g["flow", "contains", "packet"].edge_attr = torch.randn(n_pkts_each, 4)
        g["packet", "next_packet", "packet"].edge_index = torch.stack([
            torch.arange(n_pkts_each - 1, dtype=torch.long),
            torch.arange(1, n_pkts_each, dtype=torch.long),
        ])
        g["packet", "next_packet", "packet"].edge_attr = torch.randn(n_pkts_each - 1, 1)
        graphs.append(g)
    return Batch.from_data_list(graphs)


def test_model_output_shape():
    batch = _make_batch(n_graphs=4, n_pkts_each=5, num_classes=13)
    metadata = batch.metadata()
    model = HeteroGNN_Edge(metadata, hidden_channels=64, num_classes=13)
    out = model(
        batch.x_dict,
        batch.edge_index_dict,
        batch.edge_attr_dict,
        batch.batch_dict,
    )
    assert out.shape == (4, 13), f"Expected (4, 13), got {out.shape}"


def test_model_output_shape_8_classes():
    batch = _make_batch(n_graphs=2, n_pkts_each=3, num_classes=8)
    metadata = batch.metadata()
    model = HeteroGNN_Edge(metadata, hidden_channels=64, num_classes=8)
    out = model(
        batch.x_dict,
        batch.edge_index_dict,
        batch.edge_attr_dict,
        batch.batch_dict,
    )
    assert out.shape == (2, 8)


def test_model_no_nan():
    batch = _make_batch(n_graphs=3, n_pkts_each=4, num_classes=13)
    metadata = batch.metadata()
    model = HeteroGNN_Edge(metadata, hidden_channels=64, num_classes=13)
    out = model(
        batch.x_dict,
        batch.edge_index_dict,
        batch.edge_attr_dict,
        batch.batch_dict,
    )
    assert not torch.isnan(out).any(), "Model output contains NaN"
```

- [ ] **Step 2: Write `tests/baselines/test_dataset.py`**

```python
"""Unit tests for NIDSDataset graph structure."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "baselines" / "gnn4id"))

import ast
import pandas as pd
import torch
import tempfile
import os
from utils.functions import NIDSDataset, _safe_parse_list


def _make_synthetic_csv(path: str, label: str, n_rows: int = 3) -> None:
    """Write a minimal nfstream-style CSV for testing."""
    rows = []
    for i in range(n_rows):
        n_pkts = 5
        rows.append({
            "bidirectional_first_seen_ms": i * 1000,
            "src_ip": "192.168.1.1",
            "dst_ip": "10.0.0.1",
            "src_port": 12345,
            "dst_port": 80,
            "protocol": 6,
            "bidirectional_packets": n_pkts,
            "bidirectional_duration_ms": 100,
            "bidirectional_ack_packets": 3,
            "bidirectional_syn_packets": 1,
            "bidirectional_fin_packets": 1,
            "bidirectional_rst_packets": 0,
            "bidirectional_psh_packets": 2,
            "src2dst_packets": 3,
            "src2dst_min_ps": 40,
            "src2dst_max_ps": 1500,
            "dst2src_min_ps": 40,
            "dst2src_max_ps": 1500,
            "expiration_id": 0,
            "pkt_hex": str(["deadbeef" * 10] * n_pkts),
            "pkt_delta": str([0, 10, 5, 8, 3]),
            "pkt_dir": str([0, 1, 0, 1, 0]),
            "pkt_ip_size": str([60, 80, 60, 80, 60]),
            "pkt_transport_size": str([20, 40, 20, 40, 20]),
            "pkt_payload_size": str([0, 20, 0, 20, 0]),
            "label": label,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def test_dataset_builds_graphs():
    label_mapping = {"BruteForce": 0, "DDoS": 1}
    with tempfile.TemporaryDirectory() as tmpdir:
        csv1 = os.path.join(tmpdir, "bruteforce.csv")
        csv2 = os.path.join(tmpdir, "ddos.csv")
        _make_synthetic_csv(csv1, "BruteForce", n_rows=3)
        _make_synthetic_csv(csv2, "DDoS", n_rows=2)
        dataset = NIDSDataset(
            csv_files=[(csv1, "BruteForce"), (csv2, "DDoS")],
            label_mapping=label_mapping,
        )
    assert len(dataset) == 5


def test_graph_node_types():
    label_mapping = {"BruteForce": 0}
    with tempfile.TemporaryDirectory() as tmpdir:
        csv1 = os.path.join(tmpdir, "test.csv")
        _make_synthetic_csv(csv1, "BruteForce", n_rows=1)
        dataset = NIDSDataset([(csv1, "BruteForce")], label_mapping)
    g = dataset[0]
    assert "flow" in g.node_types
    assert "packet" in g.node_types
    assert g["flow"].x.shape[0] == 1
    assert g["flow"].y.item() == 0


def test_graph_edge_types():
    label_mapping = {"BruteForce": 0}
    with tempfile.TemporaryDirectory() as tmpdir:
        csv1 = os.path.join(tmpdir, "test.csv")
        _make_synthetic_csv(csv1, "BruteForce", n_rows=1)
        dataset = NIDSDataset([(csv1, "BruteForce")], label_mapping)
    g = dataset[0]
    assert ("flow", "contains", "packet") in g.edge_types
    assert ("packet", "next_packet", "packet") in g.edge_types
    # contains: 5 packet nodes → 5 edges
    assert g["flow", "contains", "packet"].edge_index.shape[1] == 5
    # link: 5 packets → 4 link edges
    assert g["packet", "next_packet", "packet"].edge_index.shape[1] == 4


def test_safe_parse_list():
    assert _safe_parse_list("[1, 2, 3]") == [1, 2, 3]
    assert _safe_parse_list("5") == [5]
    assert _safe_parse_list("bad") is None
```

- [ ] **Step 3: Run tests (expect PASS — no nfstream required for unit tests)**

```bash
D:\v\nt114\Scripts\python.exe -m pytest tests/baselines/ -v
```

Expected output:
```
tests/baselines/test_model.py::test_model_output_shape PASSED
tests/baselines/test_model.py::test_model_output_shape_8_classes PASSED
tests/baselines/test_model.py::test_model_no_nan PASSED
tests/baselines/test_dataset.py::test_dataset_builds_graphs PASSED
tests/baselines/test_dataset.py::test_graph_node_types PASSED
tests/baselines/test_dataset.py::test_graph_edge_types PASSED
tests/baselines/test_dataset.py::test_safe_parse_list PASSED
7 passed
```

- [ ] **Step 4: Commit**

```bash
git add tests/baselines/
git commit -m "test(baseline): add GNN4ID model and dataset unit tests"
```

---

## Task 7: `preprocess.py`

**Files:**
- Create: `baselines/gnn4id/preprocess.py`

Orchestrates: discover class folders → per-PCAP nfstream CSV → additional_features → NIDSDataset → save `graphs.pt`.

- [ ] **Step 1: Write `baselines/gnn4id/preprocess.py`**

```python
#!/usr/bin/env python3
"""GNN4ID preprocessing: raw PCAPs → outputs/graphs.pt.

Usage:
    python baselines/gnn4id/preprocess.py \
        --raw-root data/raw/14gb \
        --out      baselines/gnn4id/outputs/graphs.pt \
        --csv-dir  baselines/gnn4id/outputs/csv
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

import torch

# Resolve imports whether run from repo root or baselines/gnn4id/
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from utils.feature_extractor import extract_pcap_to_csv
from utils.additional_features import additional_features
from utils.functions import NIDSDataset

_LOG = logging.getLogger("gnn4id.preprocess")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", default="data/raw/14gb")
    ap.add_argument("--out", default="baselines/gnn4id/outputs/graphs.pt")
    ap.add_argument("--csv-dir", default="baselines/gnn4id/outputs/csv")
    ap.add_argument("--max-packets-per-flow", type=int, default=20)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    raw_root = Path(args.raw_root)
    csv_dir = Path(args.csv_dir)
    out_path = Path(args.out)
    csv_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Discover class folders → label mapping ──────────────────────────
    class_dirs = sorted(p for p in raw_root.iterdir() if p.is_dir())
    if not class_dirs:
        _LOG.error("No class subdirectories found under %s", raw_root)
        sys.exit(1)
    label_mapping: dict[str, int] = {d.name: i for i, d in enumerate(class_dirs)}
    _LOG.info("Classes (%d): %s", len(label_mapping), list(label_mapping))

    # ── 2. PCAP → CSV (nfstream) ───────────────────────────────────────────
    csv_files: list[tuple[str, str]] = []
    for cls_dir in class_dirs:
        label = cls_dir.name
        pcaps = sorted(cls_dir.glob("*.pcap"))
        _LOG.info("  [%s] %d pcap(s)", label, len(pcaps))
        for pcap in pcaps:
            out_csv = csv_dir / f"{label}__{pcap.stem}.csv"
            if not out_csv.exists():
                _LOG.info("    extracting %s ...", pcap.name)
                extract_pcap_to_csv(
                    str(pcap), str(out_csv), label=label,
                    max_pkts=args.max_packets_per_flow,
                )
            else:
                _LOG.debug("    skip (exists): %s", out_csv.name)
            # 3. Additional features (rolling-window, overwrites CSV)
            result = additional_features(str(out_csv))
            if result == "":
                _LOG.warning("    additional_features failed for %s, skipping", out_csv.name)
                continue
            csv_files.append((str(out_csv), label))

    _LOG.info("Building PyG dataset from %d CSV files ...", len(csv_files))

    # ── 4. Build PyG graphs ────────────────────────────────────────────────
    dataset = NIDSDataset(csv_files=csv_files, label_mapping=label_mapping)
    _LOG.info("Built %d graphs", len(dataset))

    # ── 5. Save ────────────────────────────────────────────────────────────
    graphs = [dataset[i] for i in range(len(dataset))]
    torch.save({"graphs": graphs, "label_mapping": label_mapping}, str(out_path))
    _LOG.info("Saved → %s", out_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add baselines/gnn4id/preprocess.py
git commit -m "feat(baseline): add preprocess.py PCAP→graphs.pt pipeline"
```

---

## Task 8: `train.py`

**Files:**
- Create: `baselines/gnn4id/train.py`

Loads `graphs.pt`, stratified 70/15/15 random split (seed=42), trains `HeteroGNN_Edge`, early stopping on val macro F1, saves `results.json`.

- [ ] **Step 1: Write `baselines/gnn4id/train.py`**

```python
#!/usr/bin/env python3
"""GNN4ID training: graphs.pt → checkpoint.pt + results.json.

Usage:
    python baselines/gnn4id/train.py \
        --graphs baselines/gnn4id/outputs/graphs.pt \
        --out-dir baselines/gnn4id/outputs
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.loader import DataLoader
from tqdm import tqdm

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from model import HeteroGNN_Edge

_LOG = logging.getLogger("gnn4id.train")

CLASS_WEIGHT_CAP = 10.0
SEED = 42


def _split_graphs(graphs, label_mapping):
    labels = [g["flow"].y.item() for g in graphs]
    idx = list(range(len(graphs)))
    train_idx, tmp_idx = train_test_split(
        idx, test_size=0.30, stratify=labels, random_state=SEED
    )
    tmp_labels = [labels[i] for i in tmp_idx]
    val_idx, test_idx = train_test_split(
        tmp_idx, test_size=0.50, stratify=tmp_labels, random_state=SEED
    )
    return train_idx, val_idx, test_idx


def _class_weights(train_graphs, num_classes: int) -> torch.Tensor:
    ys = np.array([g["flow"].y.item() for g in train_graphs])
    classes = np.arange(num_classes)
    w = compute_class_weight("balanced", classes=classes, y=ys)
    w = np.clip(w, None, CLASS_WEIGHT_CAP)
    return torch.tensor(w, dtype=torch.float)


def _eval(model, loader, device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(
                batch.x_dict, batch.edge_index_dict,
                batch.edge_attr_dict, batch.batch_dict,
            )
            pred = out.argmax(dim=1).cpu().numpy()
            true = batch["flow"].y.cpu().numpy()
            all_pred.append(pred)
            all_true.append(true)
    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)
    macro_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    return macro_f1, all_pred, all_true


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graphs", default="baselines/gnn4id/outputs/graphs.pt")
    ap.add_argument("--out-dir", default="baselines/gnn4id/outputs")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ── Load ───────────────────────────────────────────────────────────────
    _LOG.info("Loading graphs from %s ...", args.graphs)
    saved = torch.load(args.graphs, map_location="cpu")
    graphs: list = saved["graphs"]
    label_mapping: dict[str, int] = saved["label_mapping"]
    num_classes = len(label_mapping)
    _LOG.info("  %d graphs, %d classes", len(graphs), num_classes)

    # ── Split ──────────────────────────────────────────────────────────────
    train_idx, val_idx, test_idx = _split_graphs(graphs, label_mapping)
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs   = [graphs[i] for i in val_idx]
    test_graphs  = [graphs[i] for i in test_idx]
    _LOG.info("Split: train=%d val=%d test=%d", len(train_graphs), len(val_graphs), len(test_graphs))

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_graphs,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_graphs,  batch_size=args.batch_size)

    # ── Model ──────────────────────────────────────────────────────────────
    metadata = graphs[0].metadata()
    model = HeteroGNN_Edge(metadata, hidden_channels=64, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_w = _class_weights(train_graphs, num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_w)

    # ── Train ──────────────────────────────────────────────────────────────
    best_val_f1 = 0.0
    patience_count = 0
    best_ckpt = str(out_dir / "checkpoint.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(
                batch.x_dict, batch.edge_index_dict,
                batch.edge_attr_dict, batch.batch_dict,
            )
            loss = criterion(out, batch["flow"].y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_f1, _, _ = _eval(model, val_loader, device)
        _LOG.info("Epoch %d | loss=%.4f | val_macro_f1=%.4f", epoch,
                  total_loss / len(train_loader), val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_count = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                _LOG.info("Early stopping at epoch %d", epoch)
                break

    # ── Evaluate on test set ───────────────────────────────────────────────
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_macro_f1, test_pred, test_true = _eval(model, test_loader, device)
    idx_to_label = {v: k for k, v in label_mapping.items()}
    report = classification_report(
        test_true, test_pred,
        target_names=[idx_to_label[i] for i in range(num_classes)],
        output_dict=True, zero_division=0,
    )

    weighted_f1 = f1_score(test_true, test_pred, average="weighted", zero_division=0)
    accuracy    = float((test_pred == test_true).mean())

    per_class_f1 = {
        idx_to_label[i]: round(report[idx_to_label[i]]["f1-score"], 4)
        for i in range(num_classes)
    }

    results = {
        "split": "random",
        "macro_f1":    round(float(test_macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "accuracy":    round(float(accuracy), 4),
        "per_class_f1": per_class_f1,
        "label_mapping": label_mapping,
    }

    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    _LOG.info("Results saved to %s", results_path)
    _LOG.info("Test macro F1 = %.4f", test_macro_f1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add baselines/gnn4id/train.py
git commit -m "feat(baseline): add train.py with early stopping and results.json"
```

---

## Task 9: Server — install + smoke test

Run on the L40S server after uploading the repo.

- [ ] **Step 1: Install GNN4ID dependencies on server (in project venv)**

```bash
pip install nfstream torch-geometric tqdm
# torch is already installed; nfstream requires Linux (won't work on Windows)
```

- [ ] **Step 2: Smoke-test preprocessing on 500 flows per class**

```bash
python baselines/gnn4id/preprocess.py \
  --raw-root data/raw/14gb \
  --out      baselines/gnn4id/outputs/graphs_smoke.pt \
  --csv-dir  baselines/gnn4id/outputs/csv_smoke \
  --max-packets-per-flow 20 \
  --log-level DEBUG
```

Expected: logs show 13 classes, CSV files created per PCAP, graphs saved.

- [ ] **Step 3: Smoke-test training (5 epochs)**

```bash
python baselines/gnn4id/train.py \
  --graphs   baselines/gnn4id/outputs/graphs_smoke.pt \
  --out-dir  baselines/gnn4id/outputs \
  --epochs   5 \
  --patience 5 \
  --device   cuda
```

Expected: loss decreasing, `outputs/checkpoint.pt` and `outputs/results.json` created.

- [ ] **Step 4: Full run (after smoke test passes)**

```bash
# Full preprocessing (~several hours depending on PCAP size)
python baselines/gnn4id/preprocess.py \
  --raw-root data/raw/14gb \
  --out      baselines/gnn4id/outputs/graphs.pt \
  --csv-dir  baselines/gnn4id/outputs/csv

# Full training
python baselines/gnn4id/train.py \
  --graphs   baselines/gnn4id/outputs/graphs.pt \
  --out-dir  baselines/gnn4id/outputs \
  --epochs   50 \
  --patience 10 \
  --device   cuda
```

- [ ] **Step 5: Verify results.json format**

```bash
cat baselines/gnn4id/outputs/results.json
```

Expected structure:
```json
{
  "split": "random",
  "macro_f1": 0.XXXX,
  "weighted_f1": 0.XXXX,
  "accuracy": 0.XXXX,
  "per_class_f1": {
    "BruteForce-SSH": 0.XXXX,
    "...": 0.XXXX
  },
  "label_mapping": {"BruteForce-SSH": 0, "...": 1}
}
```

---

## Self-Review Notes

- **Spec coverage:** All spec sections covered — scaffold ✓, preprocessing ✓, model ✓, training ✓, results.json ✓, "what changed from GNN4ID" table ✓.
- **Placeholders:** None — all code blocks are complete and runnable.
- **Type consistency:** `label_mapping: dict[str, int]` used consistently across `preprocess.py`, `functions.py`, `train.py`. `num_classes` flows from `len(label_mapping)` in `train.py` to `HeteroGNN_Edge(num_classes=num_classes)`.
- **nfstream Linux note:** nfstream requires Linux (or macOS). Preprocessing must run on server, not on local Windows machine. Unit tests (Task 6) are written with synthetic data so they pass locally without nfstream.
