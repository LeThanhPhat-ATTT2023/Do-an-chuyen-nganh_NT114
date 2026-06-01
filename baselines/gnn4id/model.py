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
