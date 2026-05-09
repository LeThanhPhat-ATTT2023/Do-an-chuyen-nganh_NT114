from __future__ import annotations

import math

import torch
from torch import nn


EdgeKey = tuple[str, str, str]


def edge_key_to_name(edge_key: EdgeKey) -> str:
    return "__".join(edge_key)


def _edge_softmax_by_dst(
    scores: torch.Tensor,
    dst_index: torch.Tensor,
    num_dst_nodes: int,
) -> torch.Tensor:
    """Softmax edge scores over incoming edges for each destination node."""
    if scores.numel() == 0:
        return scores

    num_heads = int(scores.shape[1])
    expanded_index = dst_index[:, None].expand(-1, num_heads)
    max_scores = torch.full(
        (num_dst_nodes, num_heads),
        -torch.inf,
        dtype=scores.dtype,
        device=scores.device,
    )

    if hasattr(max_scores, "scatter_reduce_"):
        max_scores.scatter_reduce_(0, expanded_index, scores, reduce="amax", include_self=True)
    else:
        for edge_idx in range(scores.shape[0]):
            dst = int(dst_index[edge_idx])
            max_scores[dst] = torch.maximum(max_scores[dst], scores[edge_idx])

    shifted = scores - max_scores[dst_index]
    exp_scores = torch.exp(shifted)
    denom = torch.zeros(
        (num_dst_nodes, num_heads),
        dtype=scores.dtype,
        device=scores.device,
    )
    denom.scatter_add_(0, expanded_index, exp_scores)
    return exp_scores / denom[dst_index].clamp_min(1e-12)


class HGTLayer(nn.Module):
    """A compact HGT-style relation-aware attention layer implemented with PyTorch ops."""

    def __init__(
        self,
        node_types: list[str],
        edge_types: list[EdgeKey],
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.2,
        ffn_multiplier: int = 2,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")

        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.query = nn.ModuleDict({node_type: nn.Linear(hidden_dim, hidden_dim) for node_type in node_types})
        self.key = nn.ModuleDict({node_type: nn.Linear(hidden_dim, hidden_dim) for node_type in node_types})
        self.value = nn.ModuleDict({node_type: nn.Linear(hidden_dim, hidden_dim) for node_type in node_types})
        self.out = nn.ModuleDict({node_type: nn.Linear(hidden_dim, hidden_dim) for node_type in node_types})

        self.relation_key = nn.ModuleDict()
        self.relation_value = nn.ModuleDict()
        self.relation_prior = nn.ParameterDict()
        for edge_type in edge_types:
            relation_name = edge_key_to_name(edge_type)
            self.relation_key[relation_name] = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.relation_value[relation_name] = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.relation_prior[relation_name] = nn.Parameter(torch.ones(num_heads))

        self.norm_attn = nn.ModuleDict({node_type: nn.LayerNorm(hidden_dim) for node_type in node_types})
        self.norm_ffn = nn.ModuleDict({node_type: nn.LayerNorm(hidden_dim) for node_type in node_types})
        self.ffn = nn.ModuleDict(
            {
                node_type: nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * ffn_multiplier),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * ffn_multiplier, hidden_dim),
                )
                for node_type in node_types
            }
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[EdgeKey, torch.Tensor],
        edge_weight_dict: dict[EdgeKey, torch.Tensor] | None = None,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[EdgeKey, torch.Tensor]]:
        aggregated: dict[str, torch.Tensor] = {
            node_type: torch.zeros(
                x.shape[0],
                self.num_heads,
                self.head_dim,
                dtype=x.dtype,
                device=x.device,
            )
            for node_type, x in x_dict.items()
        }
        attention_dict: dict[EdgeKey, torch.Tensor] = {}

        for edge_type in self.edge_types:
            src_type, _, dst_type = edge_type
            edge_index = edge_index_dict[edge_type]
            if edge_index.numel() == 0:
                continue

            src_index = edge_index[0].long()
            dst_index = edge_index[1].long()
            relation_name = edge_key_to_name(edge_type)

            src_x = x_dict[src_type]
            dst_x = x_dict[dst_type]
            q = self.query[dst_type](dst_x)[dst_index].view(-1, self.num_heads, self.head_dim)
            k = self.key[src_type](src_x)[src_index]
            v = self.value[src_type](src_x)[src_index]

            k = self.relation_key[relation_name](k).view(-1, self.num_heads, self.head_dim)
            v = self.relation_value[relation_name](v).view(-1, self.num_heads, self.head_dim)

            scores = (q * k).sum(dim=-1) / math.sqrt(self.head_dim)
            scores = scores * self.relation_prior[relation_name].view(1, self.num_heads)
            if edge_weight_dict is not None and edge_type in edge_weight_dict:
                edge_weight = edge_weight_dict[edge_type].to(dtype=scores.dtype, device=scores.device)
                if edge_weight.ndim == 2:
                    edge_weight = edge_weight[:, 0]
                scores = scores + edge_weight.clamp_min(1e-6).log().view(-1, 1)
            attn = _edge_softmax_by_dst(scores, dst_index, int(dst_x.shape[0]))
            messages = attn.unsqueeze(-1) * v
            if return_attention:
                attention_dict[edge_type] = attn.detach().mean(dim=1)

            scatter_index = dst_index[:, None, None].expand(-1, self.num_heads, self.head_dim)
            aggregated[dst_type].scatter_add_(0, scatter_index, messages)

        next_x: dict[str, torch.Tensor] = {}
        for node_type, x in x_dict.items():
            message = aggregated[node_type].reshape(x.shape[0], self.hidden_dim)
            attended = self.out[node_type](message)
            h = self.norm_attn[node_type](x + self.dropout(attended))
            h = self.norm_ffn[node_type](h + self.dropout(self.ffn[node_type](h)))
            next_x[node_type] = h
        if return_attention:
            return next_x, attention_dict
        return next_x


class HeteroGraphTransformer(nn.Module):
    """HGT-style flow classifier for the three-tier IDS graph artifact."""

    def __init__(
        self,
        node_input_dims: dict[str, int],
        edge_types: list[EdgeKey],
        num_classes: int,
        num_tactics: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        ffn_multiplier: int = 2,
    ) -> None:
        super().__init__()
        node_types = ["flow", "packet", "technique", "tactic"]
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.num_tactics = num_tactics

        self.input_projection = nn.ModuleDict(
            {
                node_type: nn.Sequential(
                    nn.Linear(node_input_dims[node_type], hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for node_type in ["flow", "packet", "technique"]
            }
        )
        self.tactic_embedding = nn.Embedding(max(num_tactics, 1), hidden_dim)

        self.layers = nn.ModuleList(
            [
                HGTLayer(
                    node_types=node_types,
                    edge_types=edge_types,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    ffn_multiplier=ffn_multiplier,
                )
                for _ in range(num_layers)
            ]
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode(
        self,
        node_features: dict[str, torch.Tensor],
        edge_index_dict: dict[EdgeKey, torch.Tensor],
        edge_weight_dict: dict[EdgeKey, torch.Tensor] | None = None,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], dict[EdgeKey, torch.Tensor]]:
        x_dict = {
            "flow": self.input_projection["flow"](node_features["flow"].float()),
            "packet": self.input_projection["packet"](node_features["packet"].float()),
            "technique": self.input_projection["technique"](node_features["technique"].float()),
        }

        tactic_count = int(node_features["tactic"].shape[0])
        tactic_index = torch.arange(
            tactic_count,
            dtype=torch.long,
            device=node_features["flow"].device,
        )
        if tactic_count == 0:
            tactic_index = torch.zeros(0, dtype=torch.long, device=node_features["flow"].device)
        x_dict["tactic"] = self.tactic_embedding(tactic_index.clamp_max(max(self.num_tactics - 1, 0)))

        layer_attention: list[dict[EdgeKey, torch.Tensor]] = []
        for layer in self.layers:
            if return_attention:
                x_dict, attention_dict = layer(
                    x_dict,
                    edge_index_dict,
                    edge_weight_dict=edge_weight_dict,
                    return_attention=True,
                )
                layer_attention.append(attention_dict)
            else:
                x_dict = layer(x_dict, edge_index_dict, edge_weight_dict=edge_weight_dict)
        if return_attention:
            merged_attention: dict[EdgeKey, torch.Tensor] = {}
            for edge_type in self.edge_types:
                tensors = [
                    attention_dict[edge_type]
                    for attention_dict in layer_attention
                    if edge_type in attention_dict
                ]
                if tensors:
                    merged_attention[edge_type] = torch.stack(tensors, dim=0).mean(dim=0)
            return x_dict, merged_attention
        return x_dict

    def forward(
        self,
        node_features: dict[str, torch.Tensor],
        edge_index_dict: dict[EdgeKey, torch.Tensor],
        edge_weight_dict: dict[EdgeKey, torch.Tensor] | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[EdgeKey, torch.Tensor]]:
        if return_attention:
            x_dict, attention_dict = self.encode(
                node_features,
                edge_index_dict,
                edge_weight_dict=edge_weight_dict,
                return_attention=True,
            )
            return self.classifier(x_dict["flow"]), attention_dict
        x_dict = self.encode(node_features, edge_index_dict, edge_weight_dict=edge_weight_dict)
        return self.classifier(x_dict["flow"])
