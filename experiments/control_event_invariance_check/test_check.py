import json
import numpy as np
import pandas as pd
import pytest
from check import HERE,ROOT,OUT,INPUT_KEYS,split,merge,equal_function,fixed_grid,state_errors,file_hash

def test_equivalence_covers_entire_interval_not_only_samples():
    u=np.array([.5,0,0,0]);other=u.copy();other[1]=.2
    a=[(u,20)];b=split(a,4);assert equal_function(a,b)==4
    with pytest.raises(AssertionError):equal_function(a,[(u,9),(other,1),(u,10)])
    with pytest.raises(AssertionError):equal_function(a,[(u,19)])

def test_grid_keeps_true_non_grid_discontinuities_and_removes_redundancy():
    u=np.array([.5,0,0,0]);v=np.array([.6,0,0,0]);a=[(u,13),(v,8)]
    x=fixed_grid(a,5);y=fixed_grid(split(a,4),5)
    assert [d for u,d in x]==[5,5,3,2,5,1]
    assert [d for u,d in x]==[d for u,d in y]
    np.testing.assert_array_equal(np.stack([u for u,d in x]),np.stack([u for u,d in y]));equal_function(a,x)
    c=merge(split(a,4));assert [d for u,d in c]==[13,8]

def test_source_only_selected_validation_windows_and_frozen_code():
    w=pd.read_csv(OUT/'windows.csv');assert len(w)==8 and w.window_id.nunique()==8
    meta=pd.read_parquet(ROOT/'experiments/control_response_eval_v1/results/windows_1000ms.parquet')
    for row in w.itertuples():
        source=meta.iloc[row.source_index];assert source.window_id==row.window_id
        assert source.split=='validation' and source.sampling_group=='main_50hz'
    m=json.loads((OUT/'manifest.json').read_text());assert not m['test_loaded'] and not m['retrained']
    assert file_hash(ROOT/'src/system_identification/models/velocity_rate_gru.py')==m['model_code_sha256']
    for path,digest in m['checkpoints'].items():assert file_hash(ROOT/path)==digest

@pytest.fixture(scope='module')
def data():return np.load(OUT/'traces.npz')

def test_identical_inputs_repeated_inference_and_fixed_grid(data):
    for uid in pd.read_csv(OUT/'windows.csv').window:
        for key in INPUT_KEYS:np.testing.assert_array_equal(data[f'{uid}__fixed_native__input_{key}'],data[f'{uid}__fixed_split4__input_{key}'])
        for seed in [17,29,43]:
            np.testing.assert_array_equal(data[f'{uid}__{seed}__native__pred'],data[f'{uid}__{seed}__repeat__pred'])
            np.testing.assert_array_equal(data[f'{uid}__{seed}__fixed_native__pred'],data[f'{uid}__{seed}__fixed_split4__pred'])
            np.testing.assert_allclose(data[f'{uid}__{seed}__native__pred'],data[f'{uid}__{seed}__pad_only__pred'],atol=2e-6,rtol=2e-6)

def test_causal_phase_initial_hidden_and_dt_unchanged(data):
    for uid in pd.read_csv(OUT/'windows.csv').window:
        for variant in ['split2','split4','merged','prior_common','prior_refined','donor','fixed_native','fixed_split4','fixed_donor','pad_only']:
            for key in INPUT_KEYS:
                if key in ['events','event_length']:continue
                np.testing.assert_array_equal(data[f'{uid}__native__input_{key}'],data[f'{uid}__{variant}__input_{key}'])
            for seed in [17,29,43]:
                for key in ['history_hidden','phase']:
                    np.testing.assert_array_equal(data[f'{uid}__{seed}__native__{key}'],data[f'{uid}__{seed}__{variant}__{key}'])

def test_divergence_starts_in_event_encoder_before_state_features():
    d=pd.read_csv(OUT/'layers.csv').query('variant=="split4"')
    for key,g in d.groupby(['window','seed']):
        x=g.set_index('layer');assert x.loc['event_encoded','first_changed_step']==1
        assert x.loc['step_hidden_out','first_changed_step']==1
        assert x.loc['features','first_changed_step']==2
        assert x.loc['step_hidden_in','first_changed_step']==2
    e=pd.read_csv(OUT/'equivalence.csv');assert e.exact_zoh_equivalent.all()
    assert e.control_50hz_mean_max_difference.max()<1e-14

# Independent raw-vector and Hamilton quaternion recomputation of every saved metric.
def test_independent_metric_recomputation(data):
    d=pd.read_csv(OUT/'metrics.csv')
    for (uid,seed,variant,ref),g in d.groupby(['window','seed','variant','reference']):
        a=data[f'{uid}__{seed}__{variant}__pred'][0].astype(float);b=data[f'{uid}__{seed}__{ref}__pred'][0].astype(float)
        times=np.cumsum(data[f'{uid}__native__input_dt'][0])
        for row in g.itertuples():
            k=int(np.argmin(abs(times-row.horizon_ms/1000)))
            if row.metric=='attitude_deg':
                q=a[:,6:10];r=b[:,6:10];q=q/np.linalg.norm(q,axis=1)[:,None];r=r/np.linalg.norm(r,axis=1)[:,None]
                scalar=np.sum(q*r,axis=1)
                vector=q[:,:1]*r[:,1:]-r[:,:1]*q[:,1:]-np.cross(q[:,1:],r[:,1:])
                values=np.rad2deg(2*np.arctan2(np.linalg.norm(vector,axis=1),np.abs(scalar)))
            else:
                sl={'position':slice(0,3),'velocity':slice(3,6),'rate':slice(10,13)}[row.metric]
                values=np.sqrt(np.sum((a[:,sl]-b[:,sl])**2,axis=1))
            assert abs(values[k]-row.difference)<1e-10
            assert abs(values[:k+1].max()-row.max_through_horizon)<1e-10
