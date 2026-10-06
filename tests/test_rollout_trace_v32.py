import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rollout_trace_v32 import trace,accumulated_rates,VARIANTS
from system_identification.models.actuator_command_gru import ActuatorCommandGRU


def case():
    torch.set_num_threads(1);torch.manual_seed(23)
    stats={}
    for key,n in [('state',14),('reference',5),('actuator',4)]:
        stats[key+'_mean']=np.zeros(n);stats[key+'_scale']=np.ones(n)
    stats.update(rate_scale=np.ones(3),velocity_scale=np.ones(3))
    m=ActuatorCommandGRU(stats)
    with torch.no_grad():m.head[-1].weight.normal_(0,.015)
    x=dict(history_state=torch.randn(2,26,14),history_command=torch.randn(2,26,4),
        history_actuator=torch.randn(2,26,4),future_command=torch.randn(2,50,4),
        initial_quaternion=torch.tensor([[1.,0,0,0]]).repeat(2,1),initial_velocity_n=torch.randn(2,3),
        initial_rates_b=torch.randn(2,3),current_mode=torch.tensor([[1.,0.]]).repeat(2,1),dt_s=torch.full((2,50),.02))
    x['history_state'][:,:,11]=3.
    with torch.no_grad():y=m(**x)
    return m,x,y


def test_exact_free_trace_and_oracles_coincide_on_model_generated_truth():
    m,x,y=case();r=trace(m,x,y)
    for key in y:np.testing.assert_array_equal(r['free_'+key],y[key].numpy())
    for variant in VARIANTS[1:]:
        for key in ['rates_b','quaternion','velocity_n','derivative','hidden','raw_head','recurrent_input']:
            np.testing.assert_array_equal(r[variant+'_'+key],r['free_'+key])


def test_oracle_uses_only_previous_truth_and_never_changes_free_hidden_path():
    m,x,y=case();before=trace(m,x,y)
    altered={k:v.clone() for k,v in y.items()};altered['rates_b'][:,10:]+=2.
    after=trace(m,x,altered)
    for key in before:
        if key.startswith('free_') or key=='encoded_hidden':np.testing.assert_array_equal(before[key],after[key])
    for variant in VARIANTS[1:]:
        key=variant+'_rates_b'
        np.testing.assert_array_equal(before[key][:,:11],after[key][:,:11])
        assert not np.array_equal(before[key][:,11:],after[key][:,11:])


def test_acceleration_accumulation_and_actual_clipping_are_recorded():
    m,x,y=case()
    with torch.no_grad():m.head[-1].weight.zero_();m.head[-1].bias.copy_(torch.tensor([7.,-7.,0.,8.,0.,-8.]))
    r=trace(m,x,y)
    np.testing.assert_array_equal(r['free_raw_head'][0,0],[7,-7,0,8,0,-8])
    np.testing.assert_array_equal(r['free_derivative'][0,0],[60,-60,0,120,0,-120])
    a=accumulated_rates(x['initial_rates_b'].numpy(),r['free_derivative'],x['dt_s'].numpy())
    np.testing.assert_allclose(a,r['free_rates_b'],rtol=2e-6,atol=1e-5)
