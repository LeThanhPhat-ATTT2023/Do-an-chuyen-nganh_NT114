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
