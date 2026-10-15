import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from test_rollout_trace_v32 import case
from system_identification.models.paper_predictors import PaperPredictor


def fixture(arm, random_output=False):
    baseline, x, _ = case()
    stats = dict(baseline.named_buffers())
    torch.manual_seed(17)
    model = PaperPredictor(stats, arm)
    if random_output:
        with torch.no_grad():
            model.head[-1].weight.normal_(0, .01)
            if arm != 'full':
                model.attitude_head[-1].weight.normal_(0, .01)
    return model, x


@pytest.mark.parametrize('arm', PaperPredictor.ARMS)
def test_zero_output_initialization_matches_physical_constant_rate(arm):
    model, x = fixture(arm)
    baseline, _, _ = case()
    torch.nn.init.zeros_(baseline.head[-1].weight)
    reference = baseline(**x)
    result = model(**x)
    for key in reference:
        torch.testing.assert_close(result[key], reference[key], rtol=0, atol=0)
    count = sum(p.numel() for p in model.parameters())
    assert abs(count / 40070 - 1) < .015


@pytest.mark.parametrize('arm', PaperPredictor.ARMS)
def test_future_command_causality_quaternion_norm_and_finite_gradients(arm):
    model, x = fixture(arm, random_output=True)
    result = model(**x)
    changed = dict(x, future_command=x['future_command'].clone())
    changed['future_command'][:, 25:] += 1
    altered = model(**changed)
    for key in result:
        torch.testing.assert_close(result[key][:, :25], altered[key][:, :25], rtol=0, atol=0)
        assert not torch.equal(result[key][:, 25:], altered[key][:, 25:])
    torch.testing.assert_close(result['quaternion'].norm(dim=-1), torch.ones(2, 50))
    sum(value.square().mean() for value in result.values()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_decoupled_has_no_cross_module_loss_gradients_but_exchanges_predictions():
    model, x = fixture('decoupled', random_output=True)
    attitude = [p for n, p in model.named_parameters() if n.startswith('attitude_')]
    velocity = [p for n, p in model.named_parameters() if not n.startswith('attitude_')]
    result = model(**x)
    gv = torch.autograd.grad(result['velocity_n'].square().sum() + result['rates_b'].square().sum(),
                             attitude, allow_unused=True, retain_graph=True)
    ga = torch.autograd.grad(result['quaternion'][:, :, 1:].square().sum(), velocity, allow_unused=True)
    assert all(g is None or g.abs().max() == 0 for g in gv + ga)
    before = {k: v.detach().clone() for k, v in result.items()}
    with torch.no_grad():
        model.attitude_head[-1].bias.add_(.2)
        after = model(**x)
    # Value coupling is deliberately retained even though gradient coupling is stopped.
    assert not torch.equal(before['rates_b'][:, 1:], after['rates_b'][:, 1:])
    torch.testing.assert_close(before['rates_b'][:, :1], after['rates_b'][:, :1], rtol=0, atol=0)


def test_multihead_retains_shared_encoder_gradients_for_both_tasks():
    model, x = fixture('multihead', random_output=True)
    result = model(**x)
    params = tuple(model.encoder.parameters())
    for i, term in enumerate([result['rates_b'].square().sum(), result['quaternion'][:, :, 1:].square().sum()]):
        gradients = torch.autograd.grad(term, params, retain_graph=i == 0)
        assert all(g.abs().sum() > 0 for g in gradients)
