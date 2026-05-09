import pytest

torch = pytest.importorskip("torch")

from graphslm_ids.models.hgt import HeteroGraphTransformer


def test_hgt_forward_tiny_graph() -> None:
    node_features = {
        "flow": torch.ones(2, 6),
        "packet": torch.ones(3, 8),
        "technique": torch.ones(2, 8),
        "tactic": torch.arange(2).view(-1, 1),
    }
    edge_index = {
        ("flow", "contains", "packet"): torch.tensor([[0, 0, 1], [0, 1, 2]]),
        ("packet", "rev_contains", "flow"): torch.tensor([[0, 1, 2], [0, 0, 1]]),
        ("packet", "matches_technique", "technique"): torch.tensor([[0, 2], [0, 1]]),
        ("technique", "rev_matches_technique", "packet"): torch.tensor([[0, 1], [0, 2]]),
        ("technique", "belongs_to_tactic", "tactic"): torch.tensor([[0, 1], [0, 1]]),
        ("tactic", "rev_belongs_to_tactic", "technique"): torch.tensor([[0, 1], [0, 1]]),
    }
    edge_weight = {
        ("packet", "matches_technique", "technique"): torch.tensor([0.9, 0.8]),
        ("technique", "rev_matches_technique", "packet"): torch.tensor([0.9, 0.8]),
    }

    model = HeteroGraphTransformer(
        node_input_dims={"flow": 6, "packet": 8, "technique": 8},
        edge_types=list(edge_index.keys()),
        num_classes=3,
        num_tactics=2,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    )

    logits = model(node_features, edge_index, edge_weight_dict=edge_weight)

    assert logits.shape == (2, 3)


def test_hgt_forward_can_return_attention() -> None:
    node_features = {
        "flow": torch.ones(1, 6),
        "packet": torch.ones(2, 8),
        "technique": torch.ones(1, 8),
        "tactic": torch.arange(1).view(-1, 1),
    }
    edge_index = {
        ("flow", "contains", "packet"): torch.tensor([[0, 0], [0, 1]]),
        ("packet", "rev_contains", "flow"): torch.tensor([[0, 1], [0, 0]]),
        ("packet", "matches_technique", "technique"): torch.tensor([[0, 1], [0, 0]]),
        ("technique", "rev_matches_technique", "packet"): torch.tensor([[0, 0], [0, 1]]),
        ("technique", "belongs_to_tactic", "tactic"): torch.tensor([[0], [0]]),
        ("tactic", "rev_belongs_to_tactic", "technique"): torch.tensor([[0], [0]]),
    }
    model = HeteroGraphTransformer(
        node_input_dims={"flow": 6, "packet": 8, "technique": 8},
        edge_types=list(edge_index.keys()),
        num_classes=3,
        num_tactics=1,
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    )

    logits, attention = model(node_features, edge_index, return_attention=True)

    assert logits.shape == (1, 3)
    assert attention[("packet", "matches_technique", "technique")].shape == (2,)
