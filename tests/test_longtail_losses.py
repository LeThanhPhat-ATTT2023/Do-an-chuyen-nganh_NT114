import torch
import torch.nn.functional as F
from graphslm_ids.offline.training.train_hgt_flow_classifier import _compute_train_loss


def test_balanced_softmax_equals_ce_on_shifted_logits():
    torch.manual_seed(0)
    logits = torch.randn(8, 4)
    labels = torch.randint(0, 4, (8,))
    log_prior = torch.log(torch.tensor([0.1, 0.2, 0.3, 0.4]))
    got = _compute_train_loss(logits, labels, weight=None,
                              loss_type="balanced_softmax", log_prior=log_prior)
    expected = F.cross_entropy(logits + log_prior, labels)
    assert torch.allclose(got, expected, atol=1e-6)


def test_balanced_softmax_uniform_prior_equals_ce():
    logits = torch.randn(8, 4)
    labels = torch.randint(0, 4, (8,))
    log_prior = torch.log(torch.full((4,), 0.25))
    got = _compute_train_loss(logits, labels, weight=None,
                              loss_type="balanced_softmax", log_prior=log_prior)
    assert torch.allclose(got, F.cross_entropy(logits, labels), atol=1e-6)


def test_ldam_margin_increases_loss_vs_plain_ce():
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    labels = torch.tensor([0, 1])
    margins = torch.tensor([0.5, 0.5, 0.5])
    ldam = _compute_train_loss(logits, labels, weight=None,
                               loss_type="ldam", ldam_margins=margins)
    ce = F.cross_entropy(logits, labels)
    assert ldam > ce


def test_ldam_rarer_class_gets_bigger_margin_helper():
    from graphslm_ids.offline.training.train_hgt_flow_classifier import ldam_margins_from_counts
    counts = torch.tensor([10000.0, 100.0])
    m = ldam_margins_from_counts(counts, max_margin=0.5)
    assert m[1] > m[0]
    assert abs(float(m.max()) - 0.5) < 1e-6
