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
        self._node_types = list(node_types)

        # Determine which node types are destinations (updated by conv)
        dst_types = {et[2] for et in edge_types}
        src_only_types = [nt for nt in node_types if nt not in dst_types]

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

        # Lazy projections for source-only node types so their pooled dim = hidden_channels
        self.src_proj = nn.ModuleDict({
            nt: nn.LazyLinear(hidden_channels) for nt in src_only_types
        })

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
        # packet.x is stored as uint8 (raw bytes 0-255) to save disk/RAM; conv
        # layers need floats, so upcast any integer feature tensor here.
        x_dict = {k: (v if torch.is_floating_point(v) else v.float())
                  for k, v in x_dict.items()}
        x0 = dict(x_dict)  # preserve original features for source-only node types
        out1 = self.conv1(x0, edge_index_dict, edge_attr_dict)
        # Merge back node types not updated by conv1 (source-only, e.g. 'flow')
        x1 = {**x0, **out1}
        x1 = {nt: F.leaky_relu(self.bn1[nt](x)) if nt in out1 else x
              for nt, x in x1.items()}

        out2 = self.conv2(x1, edge_index_dict, edge_attr_dict)
        # Merge back again for pool step
        x2 = {**x1, **out2}
        x2 = {nt: F.leaky_relu(self.bn2[nt](x)) if nt in out2 else x
              for nt, x in x2.items()}

        pooled = []
        for nt in sorted(x2):
            h = x2[nt]
            # Project source-only types to hidden_channels before pooling
            if nt in self.src_proj:
                h = F.leaky_relu(self.src_proj[nt](h))
            pooled.append(global_mean_pool(h, batch_dict[nt]))
        return self.classifier(torch.cat(pooled, dim=1))
