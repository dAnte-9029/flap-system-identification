import torch

from latent import LatentTransformerHarmonics
from hybrid import query
from test_controlled import check_control_event


def test_native_command_event_and_control_gradients():
    check_control_event(LatentTransformerHarmonics, query)


def test_dense_slow_readout_and_padding():
    torch.manual_seed(17)
    torch.set_num_threads(1)
    model = LatentTransformerHarmonics(torch.zeros(30), torch.ones(30))
    h = torch.randn(2, 26, 30)
    # Disable harmonic motion so total and slow velocities must agree.
    h[:, -1, 21:23] = 0.
    h[:, -1, 15] = 3.
    s = torch.randn(2, 13)
    s[:, 6:10] = torch.nn.functional.normalize(s[:, 6:10], dim=-1)
    u = torch.rand(2, 3, 2, 4, requires_grad=True)
    dt = u.new_tensor([7000., 13000.]).expand(2, 3, 2)
    tape = torch.cat((u, dt[..., None]), -1)
    roll = model(h, s, tape)
    t = u.new_tensor([[0., .003, .007, .017, .039, .06]]).expand(2, -1)
    v = query(roll, t, 'slow_v')
    w = query(roll, t, 'slow_w')
    torch.testing.assert_close(query(roll, t, 'pv')[..., 3:], v)
    torch.testing.assert_close(query(roll, t, 'w'), w)
    torch.testing.assert_close(v[:, 0], s[:, 3:6])
    torch.testing.assert_close(w[:, 0], s[:, 10:])
    # Dense query uses only elapsed time, independent of the first segment's eventual duration.
    later = tape.detach().clone()
    later[:, 0, :, 4] = u.new_tensor([11000., 9000.])
    delayed = model(h, s, later)
    for component in ('slow_v', 'slow_w', 'pv', 'q', 'w'):
        torch.testing.assert_close(query(delayed, t[:, :3], component), query(roll, t[:, :3], component))
    padded = torch.cat((tape, torch.rand_like(tape[:, :, :1])), 2)
    padded[:, :, -1, 4] = 0
    extended = model(h, s, padded)
    torch.testing.assert_close(query(extended, t, 'pv'), query(roll, t, 'pv'))
    torch.testing.assert_close(query(extended, t, 'w'), query(roll, t, 'w'))
    before = torch.autograd.grad(w[:, :3].sum(), u, retain_graph=True)[0]
    assert torch.count_nonzero(before[:, 0, 1:]) == 0 and torch.count_nonzero(before[:, 1:]) == 0
    grads = torch.autograd.grad(w[:, -1].sum(), (u, model.latent_init.weight, model.latent_feedback.weight,
                                                model.latent_command.weight, model.latent_target.weight, model.log_rate))
    assert all(g.isfinite().all() and g.abs().sum() > 0 for g in grads)
