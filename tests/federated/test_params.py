import pytest
import torch

from graphslm_ids.models.hgt import HeteroGraphTransformer
from graphslm_ids.offline.federated.params import ParamLayout, group_of


def _tiny_model():
    edge_types = [
        ("flow", "contains", "packet"),
        ("packet", "rev_contains", "flow"),
        ("flow", "burst_neighbor", "flow"),
    ]
    return HeteroGraphTransformer(
        node_input_dims={"flow": 4, "packet": 3, "technique": 2},
        edge_types=edge_types,
        num_classes=3,
        num_tactics=2,
        hidden_dim=8,
        num_layers=2,
        num_heads=2,
        node_types=["flow", "packet", "technique", "tactic"],
    )


def test_flatten_unflatten_round_trip():
    model = _tiny_model()
    layout = ParamLayout.from_model(model)
    vec = layout.flatten(model.state_dict())
    assert vec.shape == (layout.numel,) and vec.dtype == torch.float32
    rebuilt = layout.unflatten(vec)
    for key, value in model.state_dict().items():
        assert torch.equal(rebuilt[key], value)
    _tiny_model().load_state_dict(rebuilt)
    with pytest.raises(ValueError):
        layout.unflatten(torch.zeros(layout.numel + 1))


def test_groups_cover_relations_nodes_and_heads():
    layout = ParamLayout.from_model(_tiny_model())
    assert {
        "rel:flow__contains__packet",
        "rel:packet__rev_contains__flow",
        "rel:flow__burst_neighbor__flow",
        "node:flow",
        "node:packet",
        "classifier",
        "tactic_embedding",
    } <= set(layout.group_names)
    covered = sum(b - a for g in layout.group_names for a, b in layout.groups[g])
    assert covered == layout.numel


def test_group_of_examples():
    assert group_of("layers.3.relation_prior.flow__contains__packet") == "rel:flow__contains__packet"
    assert group_of("layers.0.relation_key.a__b__c.weight") == "rel:a__b__c"
    assert group_of("layers.0.ffn.host.0.weight") == "node:host"
    assert group_of("input_projection.packet.0.bias") == "node:packet"
    assert group_of("classifier.2.weight") == "classifier"


def test_active_groups_detect_untouched_relation():
    layout = ParamLayout.from_model(_tiny_model())
    deltas = torch.randn(2, layout.numel)
    cols = layout.group_coords(["rel:flow__burst_neighbor__flow"])
    deltas[1, cols] = 0.0
    active = layout.active_groups(deltas)
    g = layout.group_names.index("rel:flow__burst_neighbor__flow")
    assert bool(active[0, g]) and not bool(active[1, g])
    total = layout.group_norms(deltas)[0].double().pow(2).sum()
    assert total.item() == pytest.approx(deltas[0].double().pow(2).sum().item(), rel=1e-5)
