import numpy as np
import pytest
import torch

from graphslm_ids.offline.federated.aggregators import ACCEPT, REJECT, AggResult, Aggregator, FedAvg, weighted_mean
from graphslm_ids.offline.federated.server import ClientMessage, FLServer


class _Scripted(Aggregator):
    """Fixed decisions (or a selected mask); records the ctx it receives."""

    name = "scripted"

    def __init__(self, decisions=None, selected=None):
        self.decisions, self.selected, self.seen_ctx = decisions, selected, None

    def aggregate(self, deltas, weights, ctx=None):
        self.seen_ctx = ctx
        if self.decisions is not None:
            keep = torch.tensor([d != REJECT for d in self.decisions])
            delta = weighted_mean(deltas[keep], weights[keep]) if keep.any() else torch.zeros(deltas.shape[1])
            trust = np.array([0.0 if d == REJECT else 1.0 for d in self.decisions])
            return AggResult(delta, decisions=list(self.decisions), trust=trust, weights=trust * weights.numpy())
        mask = torch.as_tensor(self.selected)
        return AggResult(weighted_mean(deltas[mask], weights[mask]), selected=mask)


def _msg(cid, vec, r=1, n=1):
    return ClientMessage(cid, r, torch.tensor(vec, dtype=torch.float32), n)


def test_server_applies_weighted_fedavg_and_advances_round():
    server = FLServer(global_vec=torch.zeros(3), aggregator=FedAvg())
    report = server.step([_msg(0, [1.0, 0.0, 0.0], n=1), _msg(1, [0.0, 2.0, 0.0], n=3)])
    assert report.applied and server.round_idx == 2
    assert torch.allclose(server.global_vec, torch.tensor([0.25, 1.5, 0.0]))
    assert torch.allclose(server.prev_aggregate, torch.tensor([0.25, 1.5, 0.0]))
    assert report.decision_map() == {0: "accept", 1: "accept"}


def test_server_rejects_invalid_messages():
    server = FLServer(global_vec=torch.zeros(2), aggregator=FedAvg())
    report = server.step(
        [_msg(0, [float("nan"), 0.0]), _msg(1, [1.0, 1.0], r=5), _msg(2, [1.0, 1.0, 1.0]), _msg(3, [1.0, 1.0])]
    )
    assert report.decision_map() == {0: "invalid", 1: "invalid", 2: "invalid", 3: "accept"}
    assert torch.allclose(server.global_vec, torch.ones(2))
    ids, deltas, samples = server.last_inputs
    assert ids == [3] and tuple(deltas.shape) == (1, 2) and samples.tolist() == [1.0]


def test_server_rejects_deltas_that_overflow_float32():
    server = FLServer(global_vec=torch.zeros(2), aggregator=FedAvg())
    huge = ClientMessage(0, 1, torch.tensor([1e300, 0.0], dtype=torch.float64), 1)
    assert server.step([huge]).decision_map() == {0: "invalid"}
    assert torch.equal(server.global_vec, torch.zeros(2))


def test_server_keeps_global_when_nothing_is_valid():
    server = FLServer(global_vec=torch.ones(2), aggregator=FedAvg())
    report = server.step([_msg(0, [float("inf"), 0.0])])
    assert not report.applied and server.round_idx == 2
    assert torch.allclose(server.global_vec, torch.ones(2)) and server.prev_aggregate is None


def test_server_uses_aggregator_decisions_and_server_lr():
    agg = _Scripted(decisions=[REJECT, ACCEPT])
    server = FLServer(global_vec=torch.zeros(2), aggregator=agg, server_lr=0.5)
    report = server.step([_msg(0, [-9.0, -9.0]), _msg(1, [2.0, 2.0])])
    assert report.decision_map() == {0: "reject", 1: "accept"}
    assert torch.allclose(server.global_vec, torch.tensor([1.0, 1.0]))
    assert agg.seen_ctx["client_ids"] == [0, 1] and agg.seen_ctx["round_idx"] == 1
    assert "prev_aggregate" in agg.seen_ctx and "layout" in agg.seen_ctx


def test_server_marks_unselected_clients_rejected():
    server = FLServer(global_vec=torch.zeros(2), aggregator=_Scripted(selected=[False, True]))
    report = server.step([_msg(0, [5.0, 5.0]), _msg(1, [1.0, 1.0])])
    assert report.decision_map() == {0: "reject", 1: "accept"}
    assert torch.allclose(server.global_vec, torch.ones(2))


def test_server_does_not_apply_when_every_client_is_rejected():
    server = FLServer(global_vec=torch.ones(2), aggregator=_Scripted(decisions=[REJECT, REJECT]))
    report = server.step([_msg(0, [5.0, 5.0]), _msg(1, [1.0, 1.0])])
    assert not report.applied and server.round_idx == 2
    assert torch.allclose(server.global_vec, torch.ones(2)) and server.prev_aggregate is None


class _WrongLength(Aggregator):
    name = "wrong_length"

    def aggregate(self, deltas, weights, ctx=None):
        return AggResult(deltas.mean(0), decisions=[ACCEPT])


def test_server_rejects_aggregator_verdicts_of_the_wrong_length():
    server = FLServer(global_vec=torch.zeros(2), aggregator=_WrongLength())
    with pytest.raises(ValueError):
        server.step([_msg(0, [1.0, 1.0]), _msg(1, [2.0, 2.0])])


def test_server_reports_messages_in_arrival_order_with_trust_and_weight():
    agg = _Scripted(decisions=[ACCEPT, REJECT, ACCEPT])
    server = FLServer(global_vec=torch.zeros(2), aggregator=agg)
    report = server.step([_msg(7, [1.0, 1.0], n=2), _msg(5, [4.0, 4.0], n=3), _msg(6, [float("nan"), 0.0]), _msg(4, [3.0, 3.0], n=4)])
    assert [d.client_id for d in report.decisions] == [7, 5, 6, 4]
    by_id = {d.client_id: d for d in report.decisions}
    assert by_id[7].weight == 2.0 and by_id[7].trust == 1.0 and by_id[7].norm == pytest.approx(2**0.5)
    assert by_id[5].decision == "reject" and by_id[5].weight == 0.0 and by_id[5].trust == 0.0
    assert by_id[6].decision == "invalid" and by_id[6].norm is None and by_id[6].reason == "non_finite"


def test_server_weighting_modes():
    server = FLServer(global_vec=torch.zeros(1), aggregator=FedAvg(), weighting="uniform")
    server.step([_msg(0, [1.0], n=1), _msg(1, [3.0], n=9)])
    assert torch.allclose(server.global_vec, torch.tensor([2.0]))
    with pytest.raises(ValueError):
        FLServer(global_vec=torch.zeros(1), aggregator=FedAvg(), weighting="bogus")
