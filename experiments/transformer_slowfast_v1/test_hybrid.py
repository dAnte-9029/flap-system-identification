import torch
from hybrid import TransformerHarmonics, query
from fit import make_model
from test_controlled import check_control_event


def test_fast_head_off_recovers_transformer_and_on_preserves_prefix():
    torch.set_num_threads(1)
    mean, scale = torch.zeros(30), torch.ones(30)
    torch.manual_seed(17)
    plain = make_model('transformer', mean, scale)
    torch.manual_seed(17)
    model = TransformerHarmonics(mean, scale)
    h = torch.randn(2, 26, 30)
    h[:, -1, 15] = 3.5
    h[:, -1, 21:23] = 1.
    s = torch.rand(2, 13)
    s[:, 6:10] = torch.nn.functional.normalize(s[:, 6:10], dim=-1)
    u = torch.rand(2, 12, 4, requires_grad=True)
    z = model.rollout(h, s, u)
    assert torch.equal(s, z[:, 0])
    torch.testing.assert_close(z[:, :8], model.rollout(h, s, u[:, :7]), atol=1e-6, rtol=1e-5)
    g, a = torch.autograd.grad(z[:, 7].sum(), (u, model.initial_amplitude.weight))
    assert g[:, 7:].abs().max() == 0 and g[:, :7].abs().sum() > 0
    assert a.isfinite().all() and a.abs().sum() > 0
    with torch.no_grad():
        for head in (model.initial_amplitude, model.amplitude, model.control_amplitude):
            for param in head.parameters():
                param.zero_()
        torch.testing.assert_close(model.rollout(h, s, u), plain.rollout(h, s, u), atol=1e-6, rtol=1e-5)


def test_native_control_event():
    check_control_event(TransformerHarmonics, query)
