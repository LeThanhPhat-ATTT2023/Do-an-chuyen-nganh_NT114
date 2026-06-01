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
