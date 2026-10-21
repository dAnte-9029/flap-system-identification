"""Representation invariance, timing, masking and malformed-input boundaries."""
import numpy as np
import pytest
import torch

from system_identification.models.control_events import (
    canonicalize_control_events as canonical,
    canonicalize_control_events_numpy,
)


def sequence(rows, dtype=torch.float64):
    return torch.tensor(rows, dtype=dtype).reshape(1, -1, 5)


def test_exact_merge_idempotence_and_real_change():
    x = sequence([[.4, 0, 0, 0, .005], [.4, 0, 0, 0, .007],
                  [.6, 0, 0, 0, .008]])
    original = x.clone()
    y, n = canonical(x, torch.tensor([3]), torch.tensor([.02]))
    torch.testing.assert_close(y, sequence([[.4, 0, 0, 0, .012], [.6, 0, 0, 0, .008]]), rtol=0, atol=0)
    assert n.tolist() == [2]
    yy, nn = canonical(y, n, torch.tensor([.02]))
    assert torch.equal(y, yy) and torch.equal(n, nn) and torch.equal(x, original)


def test_near_equal_commands_are_not_merged():
    u = np.float32(.5)
    v = np.nextafter(u, np.float32(1))
    x = sequence([[u, 0, 0, 0, .01], [v, 0, 0, 0, .01]], torch.float32)
    y, n = canonical(x, torch.tensor([2]))
    assert n.item() == 2 and torch.equal(x, y)


def test_padding_zero_duration_and_empty_interval():
    x = torch.full((2, 5, 5), float('nan'), dtype=torch.float64)
    x[0, :3] = sequence([[.4, 0, 0, 0, .01], [.9, 1, 1, 1, 0], [.4, 0, 0, 0, .01]])[0]
    y, n = canonical(x, torch.tensor([3, 0]), torch.tensor([.02, 0]))
    assert n.tolist() == [1, 1]
    torch.testing.assert_close(y[0:1], sequence([[.4, 0, 0, 0, .02]]), rtol=0, atol=0)
    assert not y[1].any()


def test_no_merge_across_state_intervals_and_numpy_adapter():
    x = np.array([[[[.4, 0, 0, 0, .01], [.4, 0, 0, 0, .01]],
                   [[.4, 0, 0, 0, .005], [.4, 0, 0, 0, .015]]]], np.float32)
    y, n = canonicalize_control_events_numpy(x, np.array([[2, 2]]), np.array([[.02, .02]]))
    assert y.shape == (1, 2, 1, 5) and n.tolist() == [[1, 1]]
    np.testing.assert_array_equal(y[..., 4], np.full((1, 2, 1), .02, np.float32))


@pytest.mark.parametrize('bad', ['negative', 'nan', 'sub_us', 'wrong_total', 'length'])
def test_bad_inputs_fail_closed(bad):
    x = sequence([[.4, 0, 0, 0, .02]])
    le, dt = torch.tensor([1]), torch.tensor([.02])
    if bad == 'negative': x[..., 4] = -.02
    if bad == 'nan': x[..., 0] = float('nan')
    if bad == 'sub_us': x[..., 4] = .0200005
    if bad == 'wrong_total': dt.fill_(.03)
    if bad == 'length': le.fill_(2)
    with pytest.raises(ValueError): canonical(x, le, dt)


def test_random_integer_clock_redundant_splits_preserve_whole_function():
    rng = np.random.default_rng(9217)
    for _ in range(100):
        duration = rng.integers(1, 10000, size=10)
        u = rng.integers(-3, 4, size=(10, 4)).astype(float)
        u[1::2] = u[::2]
        old = np.c_[u, duration * 1e-6]
        pieces = []
        for command, d in zip(u, duration):
            cuts = np.unique(np.r_[0, rng.integers(0, d+1, size=9), d])
            pieces.extend([*command, t*1e-6] for t in np.diff(cuts))
        a, na = canonicalize_control_events_numpy(old[None], np.array([len(old)]))
        b, nb = canonicalize_control_events_numpy(np.array(pieces)[None], np.array([len(pieces)]))
        np.testing.assert_array_equal(a, b); np.testing.assert_array_equal(na, nb)
        # Independent evaluation on every original/canonical boundary, not a coarse grid.
        ea = np.cumsum(duration)
        eb = np.cumsum(np.rint(a[0, :, 4]*1e6).astype(int))
        assert ea[-1] == eb[-1]
        starts = np.unique(np.r_[0, ea[:-1], eb[:-1]])
        np.testing.assert_array_equal(u[np.searchsorted(ea, starts, side='right')],
                                      a[0, np.searchsorted(eb, starts, side='right'), :4])


def test_model_entry_canonicalizes_raw_inputs_and_preserves_checkpoint_layout():
    from system_identification.models.velocity_rate_gru import VelocityRateGRU
    from system_identification.models.control_events import LEGACY_CONTROL_EVENTS, CANONICAL_CONTROL_EVENTS
    from test_velocity_rate_long_horizon import inputs
    torch.set_num_threads(2); torch.manual_seed(43)
    norm=dict(mean=[0]*11,scale=[1]*11,output_scale=[.1]*6)
    m=VelocityRateGRU(norm,control=True).double().eval()
    torch.nn.init.normal_(m.head.weight,std=.02)
    old=VelocityRateGRU(norm,control=True,control_event_policy=LEGACY_CONTROL_EVENTS).double().eval()
    old.load_state_dict(m.state_dict(),strict=True)
    assert m.control_event_policy==CANONICAL_CONTROL_EVENTS
    assert sum(p.numel() for p in m.parameters())==37590
    x=inputs(); refined={k:v.clone() for k,v in x.items()}
    refined['events']=x['events'].repeat_interleave(2,dim=2)
    refined['events'][...,4]/=2;refined['event_length']*=2
    captured=[]
    hook=m.event_gru.register_forward_pre_hook(lambda module,args:captured.append(args[0].clone()))
    with torch.no_grad():
        a=m(x);b=m(refined)
        assert torch.equal(a,b) and torch.equal(captured[0],captured[1])
        ev,le=canonical(x['events'],x['event_length'],x['dt'])
        assert torch.equal(a,old(dict(x,events=ev,event_length=le)))
        assert (old(x)-old(refined)).abs().max()>1e-6
    hook.remove()


def test_packer_normalizes_publication_repeats_and_keeps_boundary_changes():
    import sys
    from pathlib import Path
    import pandas as pd
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from prepare_velocity_rate_learning import interval_events
    from system_identification.models.control_events import LEGACY_CONTROL_EVENTS
    motor=pd.DataFrame({'timestamp_us':[-10,10,20], 'control[0]':[.4,.6,.8]})
    servo=pd.DataFrame({'timestamp_us':[-10], 'control[0]':[.1], 'control[1]':[-.1], 'control[2]':[0.]})
    repeated=pd.DataFrame({'timestamp_us':[-10,3,10,14,20,25], 'control[0]':[.4,.4,.6,.6,.8,.8]})
    starts=np.array([0,20]);ends=np.array([20,40])
    a,la=interval_events(motor,servo,starts,ends)
    b,lb=interval_events(repeated,servo,starts,ends)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(la,lb)
    assert la.tolist()==[2,1]
    np.testing.assert_array_equal(a[0,:2,0],np.array([.4,.6],np.float32))
    assert a[1,0,0]==np.float32(.8)
    legacy,ll=interval_events(repeated,servo,starts,ends,event_policy=LEGACY_CONTROL_EVENTS)
    cc,cl=canonicalize_control_events_numpy(legacy,ll,(ends-starts)*1e-6)
    np.testing.assert_array_equal(a,cc);np.testing.assert_array_equal(la,cl)
    from system_identification.data.velocity_rate_dataset import action_replay
    from test_velocity_rate_learning import assert_events_match_tape
    repeated['timestamp_sample_us']=repeated.timestamp_us
    servo['timestamp_sample_us']=servo.timestamp_us
    tape=action_replay(repeated,servo,np.array([0,20,40]),160000)
    for j in range(2):assert_events_match_tape(a[j,:la[j]],tape,j,starts[j],ends[j])
    wrong=a[0,:la[0]].copy();wrong[0,4]+=1e-6;wrong[1,4]-=1e-6
    with pytest.raises(AssertionError):assert_events_match_tape(wrong,tape,0,0,20)


def test_masked_tail_is_a_zero_token_without_changing_valid_prefix():
    from system_identification.models.velocity_rate_gru import VelocityRateGRU
    from test_velocity_rate_long_horizon import inputs
    torch.manual_seed(17)
    m=VelocityRateGRU(dict(mean=[0]*11,scale=[1]*11,output_scale=[.1]*6),control=True).double().eval()
    torch.nn.init.normal_(m.head.weight,std=.02)
    x=inputs()
    with torch.no_grad():
        before=m(x)
        x['events'][0,50:]=0;x['event_length'][0,50:]=1;x['dt'][0,50:]=0
        after=m(x)
    assert torch.equal(before[0,:50],after[0,:50])
    assert torch.equal(before[1],after[1]) and torch.isfinite(after).all()


def test_model_rejects_unknown_policy():
    from system_identification.models.velocity_rate_gru import VelocityRateGRU
    with pytest.raises(ValueError,match='policy'):
        VelocityRateGRU(dict(mean=[0]*11,scale=[1]*11,output_scale=[.1]*6),control_event_policy='typo')
