import json
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.spatial.transform import Rotation
from evaluate import (HERE,OUT,ROOT,INPUT_KEYS,splice_tape,event_pack,increments,contrast,response_error,
                      flight_weights,components,VelocityRateGRU,file_hash)

def test_increment_contrast_removes_different_initial_states():
    initial=np.zeros((2,13));initial[:,6]=1;initial[:,3]=[10,5];initial[:,10]=[3,-2]
    final=initial[:,None,:].copy();final[:,:,3]+=2;final[:,:,10]+=.3
    inc=increments(final,initial)
    for key in ['velocity','rate']:
        np.testing.assert_allclose(contrast({key:inc[key][:1]},{key:inc[key][1:]},key),0,atol=1e-12)
    assert final[0,0,3]-final[1,0,3]==5

def test_spatial_so3_increment_and_quaternion_sign():
    r0=Rotation.from_euler('z',[0,40],degrees=True);dr=Rotation.from_euler('z',[10,10],degrees=True)
    a=np.zeros((2,13));b=a[:,None,:].copy();a[:,6:10]=r0.as_quat()[:,[3,0,1,2]];b[:,0,6:10]=(dr*r0).as_quat()[:,[3,0,1,2]]
    b[1,0,6:10]*=-1
    inc=increments(b,a);np.testing.assert_allclose(contrast({'attitude':inc['attitude'][:1]},{'attitude':inc['attitude'][1:]},'attitude'),0,atol=1e-12)
    np.testing.assert_allclose(response_error(np.array([[0,0,20.]]),np.array([[0,0,10.]]),'attitude'),10)

def test_splice_relative_publication_time_and_self_swap():
    own=(np.array([0,100,200,300]),np.array([[.1,0,0,0],[.2,0,0,0],[.3,0,0,0],[.4,0,0,0]]))
    donor=(np.array([1000,1100,1200]),np.array([[.6,0,0,0],[.7,0,0,0],[.8,0,0,0]]))
    t,u=splice_tape(own,donor,150,1050,200)
    np.testing.assert_array_equal(t,[0,100,150,200,300]);np.testing.assert_allclose(u[:,0],[.1,.2,.6,.7,.8])
    st,su=splice_tape(own,own,150,150,200)
    np.testing.assert_array_equal(st,own[0]);np.testing.assert_array_equal(su,own[1])

def test_event_half_open_duration_and_value_legality():
    t=np.array([0,40,100]);u=np.array([[.2,0,0,0],[.8,0,0,0],[.9,0,0,0]])
    ev=event_pack(t,u,np.array([0,100]))[0]
    np.testing.assert_allclose(ev[:,4],[40e-6,60e-6]);np.testing.assert_allclose(ev[:,0],[.2,.8])
    with pytest.raises(AssertionError):event_pack(t,u+10,np.array([0,100]))

def test_flight_weights_and_dependency_components():
    p=pd.DataFrame(dict(flight_a=['a','a','d'],flight_b=['b','c','e']))
    np.testing.assert_allclose(flight_weights(p),[.3,.3,.4])
    comp=components(p);assert comp[0]==comp[1] and comp[2]!=comp[0]

def test_frozen_cohort_partition_and_nonoverlap():
    cfg=json.loads((HERE/'config.json').read_text());p=pd.read_parquet(OUT/'cohorts.parquet')
    manifest=json.loads((OUT/'selection_manifest.json').read_text());assert manifest['future_outcomes_used'] is False
    assert file_hash(OUT/'cohorts.parquet')==manifest['cohort_sha256']
    assert set(p.split)=={'train','validation'} and set(p.group)=={'main_50hz'}
    assert (p.split_pair==p.split+'-'+p.split).all()
    for (h,cohort,split),g in p.groupby(['horizon_ms','cohort','split']):
        assert not g.pair_uid.duplicated().any();assert g.primary.all()
        source=pd.read_parquet(cfg['audit_source']+f'/results/windows_{h}ms.parquet')
        spans={}
        for row in g.itertuples():
            for ix in [row.i,row.j]:
                r=source.iloc[ix];assert r.split==split
                spans.setdefault(r.flight_id,[]).append((r.history_start_us,r.pair_support_end_us))
        for v in spans.values():
            v.sort();assert all(v[i][1]<v[i+1][0] for i in range(len(v)-1))
    signal=p.query('cohort=="signal" and split=="validation"')
    assert signal.groupby('horizon_ms').size().to_dict()=={500:21,1000:24}

def test_selected_support_within_flight_and_outside_exclusions():
    root=ROOT/'dataset/velocity_rate_multistep_v1'
    provenance={r['flight_id']:r for r in json.loads((root/'flight_provenance.json').read_text())
                if r['split'] in ['train','validation']}
    count=0
    for h in [500,1000]:
        for m in pd.read_parquet(OUT/f'windows_{h}ms.parquet').itertuples():
            p=provenance[m.flight_id];assert m.split==p['split']
            assert p['start_us']<=m.history_start_us<=m.decision_us<m.pair_support_end_us<=p['end_us']
            assert m.command_end_us==m.decision_us+h*1000
            for a,b in p['exclude_intervals_us']:
                assert m.pair_support_end_us<a or m.history_start_us>b
            count+=1
    assert count==889

def model_and_input(control=True):
    cfg=json.loads((HERE/'config.json').read_text());name='GRU-Control' if control else 'GRU-NoControl'
    ck=torch.load(ROOT/cfg['model_root']/name/'seed_17/best.pt',weights_only=False,map_location='cpu')
    model=VelocityRateGRU(ck['normalization'],control=control);model.load_state_dict(ck['model']);model.eval()
    z=np.load(OUT/'predictions_500ms.npz');data={k:torch.from_numpy(z[k][:2].copy()) for k in INPUT_KEYS}
    return model,data,z

def test_model_target_poison_and_no_control_noninterference():
    torch.set_num_threads(2);model,data,z=model_and_input(False)
    with torch.no_grad():
        a=model(data);other={k:v.clone() for k,v in data.items()}
        other['history'][:,:,13:17]=float('nan');other['events'][:,:,:,:4]=float('nan')
        other['target']=torch.full_like(a,float('nan'));b=model(other)
    torch.testing.assert_close(a,b,atol=0,rtol=0)

def test_fixed_history_hidden_state_and_masked_phase():
    model,data,z=model_and_input(True);captures=[]
    hook=model.history_gru.register_forward_hook(lambda module,args,out:captures.append(out[0].detach().clone()))
    with torch.no_grad():
        first=model(data);other={k:v.clone() for k,v in data.items()};other['events'][:,:,:,:4]*=.8;model(other)
    hook.remove();torch.testing.assert_close(captures[0],captures[1],atol=0,rtol=0)
    # Missing phase/frequency histories are masked consistently, not replaced by future truth.
    data['history_mask'][:,:,17:20]=False
    with torch.no_grad():
        data['history'][:,:,17:20]=0;a=model(data)
        data['history'][:,:,17:20]=42;b=model(data)
    torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)

def test_autonomous_state_feedback_is_predicted_state():
    model,data,z=model_and_input(True);states=[];original=model.features
    def capture(state,*args):
        states.append(state.detach().clone());return original(state,*args)
    model.features=capture
    with torch.no_grad():pred=model(data)
    # First call encodes observed history; subsequent calls use initial, then prior predictions.
    torch.testing.assert_close(states[1],data['initial'])
    for k in range(1,len(states)-1):torch.testing.assert_close(states[k+1],pred[:,k-1],atol=0,rtol=0)

def test_common_control_grid_and_no_control_swap_results():
    for h in [500,1000]:
        z=np.load(OUT/f'predictions_{h}ms.npz')
        np.testing.assert_array_equal(z['common_events'][...,4],z['common_swap_events'][...,4])
        for seed in [17,29,43]:
            key=f'GRU-NoControl_{seed}'
            np.testing.assert_allclose(z[key+'_swapped'],z[key][z['recipient']],atol=2e-6,rtol=2e-6)
            np.testing.assert_allclose(z[key+'_common'],z[key+'_common_swapped'],atol=0,rtol=0)
    th=pd.read_csv(OUT/'direction_thresholds.csv');assert th.source.eq('train small_control only').all()
    comparisons=pd.read_csv(OUT/'paired_comparisons.csv').query('cohort=="signal" and split=="validation"')
    assert (comparisons.components==3).all() and not comparisons.interval_available.any()
    assert comparisons.component_bootstrap_low.isna().all()
