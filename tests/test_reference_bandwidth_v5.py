import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from scipy.signal import lfilter,freqz
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from system_identification.models.reference_antialias_readout import antialias_taps,antialias_rate_readout
from reference_bandwidth_v5 import causal_expand_50hz,forecast_loss
from test_control_windows_v2 import model_and_input


def test_fir_readout_matches_causal_lfilter_and_preserves_prefix():
    torch.manual_seed(9);history=torch.randn(2,51,3,dtype=torch.float64);future=torch.randn(2,100,3,dtype=torch.float64)
    taps=antialias_taps(torch.float64).numpy();joint=torch.cat([history[:,-20:],future],1).numpy()
    expected=lfilter(taps,[1],joint,axis=1)[:,21::2]
    actual=antialias_rate_readout(history,future)
    np.testing.assert_allclose(actual.numpy(),expected,atol=1e-12,rtol=1e-12)
    altered=future.clone();altered[:,50:]=999
    torch.testing.assert_close(actual[:,:25],antialias_rate_readout(history,altered)[:,:25],rtol=0,atol=0)


def test_filter_has_known_delay_stopband_and_no_zero_initialization():
    taps=antialias_taps(torch.float64).numpy();np.testing.assert_allclose(taps,taps[::-1],atol=1e-15)
    assert abs(taps.sum()-1)<1e-12 and abs(np.sum(np.arange(21)*taps)*.01-.1)<1e-12
    f,h=freqz(taps,worN=32768,fs=100)
    assert abs(h[f>=25]).max()<.01 and abs(h[f<=8]).min()>.95
    actual=antialias_rate_readout(torch.ones(1,51,3)*2,torch.ones(1,100,3)*2)
    torch.testing.assert_close(actual,torch.ones(1,50,3)*2)
    with pytest.raises(ValueError):antialias_rate_readout(torch.zeros(1,19,3),torch.zeros(1,100,3))


def test_expansion_holds_previous_forecast_instead_of_looking_ahead():
    pred={'rates_b':torch.tensor([[[2.],[4.],[6.]]])};initial={'rates_b':torch.tensor([[0.]])}
    torch.testing.assert_close(causal_expand_50hz(pred,initial)['rates_b'],torch.tensor([[[0.],[2.],[2.],[4.],[4.],[6.]]]))


def test_rate_readout_loss_has_finite_model_gradients():
    model,x=model_and_input();x['dt_s'].fill_(.01)
    for arm in ['raw100','aa50']:
        model.zero_grad();p=model(**x);target={k:v.detach().clone() for k,v in p.items()};target['rates_b']+=.1
        loss=forecast_loss(p,target,model,x['history_state'][:,:,3:6],arm);loss.backward()
        assert torch.isfinite(loss) and all(t.grad is None or torch.isfinite(t.grad).all() for t in model.parameters())


def test_adapter_checks_grid_and_never_filters_attitude_integration():
    from system_identification.models.reference_bandwidth_predictor import ReferenceBandwidthPredictor
    from system_identification.models.reference_gru_unit_gravity import ReferenceGRUUnitGravity
    old,x=model_and_input();stats={k:v for k,v in old.state_dict().items() if k.endswith('_mean') or k.endswith('_scale')};model=ReferenceGRUUnitGravity(stats)
    for name in ['history_state','history_reference','history_actuator']:
        x[name]=torch.cat([x[name],x[name][:,:25]],1)
    x['dt_s'].fill_(.01)
    raw=model(**x);adapted=ReferenceBandwidthPredictor(model,'aa50')(**x)
    torch.testing.assert_close(adapted['quaternion'],raw['quaternion'][:,1::2])
    torch.testing.assert_close(adapted['rates_b'],antialias_rate_readout(x['history_state'][:,:,3:6],raw['rates_b']))
    x['dt_s'].fill_(.02)
    with pytest.raises(ValueError):ReferenceBandwidthPredictor(model,'aa50')(**x)
