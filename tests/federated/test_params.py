import re

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


def test_group_of_remaining_branches():
    assert group_of("layers.1.relation_value.a__b__c.weight") == "rel:a__b__c"
    assert group_of("tactic_embedding.weight") == "tactic_embedding"
    assert group_of("family_head.weight") == "other"
    assert group_of("family_head.bias") == "other"


def test_non_float_entry_is_rejected():
    with pytest.raises(ValueError, match="non-float"):
        ParamLayout.from_state_dict({"classifier.0.num_batches": torch.zeros(2, dtype=torch.long)})


def test_fp16_and_zero_dim_round_trip_restores_dtype():
    state = {
        "classifier.0.weight": torch.tensor([[0.5, -1.25, 3.0], [2.0, 0.0, -0.75]], dtype=torch.float16),
        "classifier.0.scale": torch.tensor(1.5, dtype=torch.float32),
        "classifier.0.half_scalar": torch.tensor(-2.5, dtype=torch.float16),
    }
    layout = ParamLayout.from_state_dict(state)
    vec = layout.flatten(state)
    assert vec.dtype == torch.float32 and vec.shape == (6 + 1 + 1,)
    rebuilt = layout.unflatten(vec)
    for key, value in state.items():
        assert rebuilt[key].dtype == value.dtype
        assert rebuilt[key].shape == value.shape
        assert torch.equal(rebuilt[key], value)


def test_group_coords_empty_is_long_tensor():
    layout = ParamLayout.from_model(_tiny_model())
    coords = layout.group_coords([])
    assert coords.dtype == torch.long and coords.numel() == 0


def _hand_layout():
    # node:flow owns two NON-adjacent ranges (0:2 and 4:6) to exercise multi-range sums.
    state = {
        "input_projection.flow.0.weight": torch.zeros(2),
        "layers.0.relation_key.a__b__c.weight": torch.zeros(2),
        "layers.0.query.flow.weight": torch.zeros(2),
        "classifier.0.weight": torch.zeros(2),
    }
    return ParamLayout.from_state_dict(state)


def test_group_norms_exact_values_against_hand_computation():
    layout = _hand_layout()
    assert layout.group_names == ["node:flow", "rel:a__b__c", "classifier"]
    deltas = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 2.0, 0.0, 0.0, 0.0],  # flow sqrt(1+4+4+0)=3, rel 5, clf 0
            [0.0, 0.0, 0.0, 12.0, 0.0, 0.0, 6.0, 8.0],  # flow 0, rel 12, clf 10
        ]
    )
    norms = layout.group_norms(deltas)
    expected = torch.tensor([[3.0, 5.0, 0.0], [0.0, 12.0, 10.0]])
    torch.testing.assert_close(norms, expected, rtol=0, atol=1e-6)
    assert norms.dtype == torch.float32
    assert layout.active_groups(deltas).tolist() == [[True, True, False], [False, True, True]]


def test_flatten_rejects_shape_mismatch_and_missing_key():
    model = _tiny_model()
    layout = ParamLayout.from_model(model)
    state = dict(model.state_dict())
    good_shape = tuple(state["classifier.4.weight"].shape)
    state["classifier.4.weight"] = torch.zeros(1)
    with pytest.raises(ValueError) as excinfo:
        layout.flatten(state)
    message = str(excinfo.value)
    assert "classifier.4.weight" in message and str((1,)) in message and str(good_shape) in message

    state = dict(model.state_dict())
    del state["classifier.4.bias"]
    with pytest.raises(ValueError, match=re.escape("classifier.4.bias")):
        layout.flatten(state)


def test_unflatten_requires_one_dimensional_vector():
    layout = ParamLayout.from_model(_tiny_model())
    with pytest.raises(ValueError, match=re.escape(str((1, layout.numel)))):
        layout.unflatten(torch.zeros(1, layout.numel))
    with pytest.raises(ValueError):
        layout.unflatten(torch.zeros(()))


@pytest.mark.parametrize("prefix", ["module.", "_orig_mod."])
def test_wrapper_prefixed_keys_are_rejected_with_a_clear_error(prefix):
    state = {prefix + k: v for k, v in _tiny_model().state_dict().items()}
    with pytest.raises(ValueError, match="prefix"):
        ParamLayout.from_state_dict(state)


def test_model_without_relations_does_not_trigger_prefix_error():
    layout = ParamLayout.from_state_dict({"classifier.0.weight": torch.zeros(3)})
    assert layout.group_names == ["classifier"]
