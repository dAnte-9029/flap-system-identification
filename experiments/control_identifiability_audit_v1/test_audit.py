import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation
from audit import (exact_difference,batch_difference,event_indices,history_errors,
                   disjoint_pairs,permitted,load_parquet,build_histories,resolve,sha,HERE)

def test_sealed_and_outcome_guards():
    with pytest.raises(ValueError,match='sealed'):permitted('test')
    with pytest.raises(ValueError,match='future outcome'):load_parquet(Path('/missing'),'samples','train',['target_velocity'])

def test_events_accumulated_not_every_publication():
    t=np.arange(11)*20000;u=np.arange(11)*.01
    np.testing.assert_array_equal(event_indices(t,u,.04,100000),[5,10])
    assert len(event_indices(t,np.zeros(11),.01,100000))==0

def test_exact_control_boundaries_and_short_event():
    a=(np.array([0,100,400,500]),np.array([[0]*4,[1]*4,[0]*4,[999]*4]))
    b=(np.array([0]),np.zeros((1,4)))
    rms,share=exact_difference(a,b,500,np.array([[.5]*4]))
    np.testing.assert_allclose(rms,np.sqrt(.6));np.testing.assert_allclose(share,.6)

def test_vectorized_integral_against_independent_scalar():
    rng=np.random.default_rng(42);thresholds=np.array([[.01]*4,[.1]*4,[.2]*4])
    for _ in range(15):
        ta=np.r_[0,np.sort(rng.choice(np.arange(1,1000),15,replace=False))]
        tb=np.r_[0,np.sort(rng.choice(np.arange(1,1000),9,replace=False))]
        ua=rng.normal(size=(len(ta),4));ub=rng.normal(size=(len(tb),4))
        expected=exact_difference((ta,ua),(tb,ub),1000,thresholds)
        tb=np.pad(tb,(0,6),constant_values=1000);ub=np.pad(ub,((0,6),(0,0)))
        got=batch_difference(ta[None],ua[None],tb[None],ub[None],1000,thresholds)
        for x,y in zip(expected,got):np.testing.assert_allclose(x,y[0],atol=1e-12)

def test_quaternion_sign_and_so3():
    x=np.zeros((1,26,15));x[:,:,3]=1;y=x.copy();y[:,:,3]=-1
    assert history_errors(x,y)[0,1]==0
    y[:,:,3:7]=Rotation.from_euler('z',20,degrees=True).as_quat()[[3,0,1,2]]
    np.testing.assert_allclose(history_errors(x,y)[0,1],20)

def test_overlap_and_flight_disjoint_count():
    meta=pd.DataFrame(dict(log_sha=['a','b','a','c','a','d'],history_start_us=[0,0,5,0,20,0],pair_support_end_us=[10,10,15,10,30,10]))
    pairs=pd.DataFrame(dict(i=[0,2,4],j=[1,3,5],score=[0,1,2]))
    assert disjoint_pairs(pairs,meta)==[0,2]
    assert disjoint_pairs(pairs,meta,True)==[0]

def test_artifact_statistics_and_boundaries():
    out=HERE/'results'
    if not (out/'run_manifest.json').exists():pytest.skip('complete run needed')
    cfg=json.loads((HERE/'config.json').read_text());root,flights,prov,m=resolve(cfg)
    summaries=pd.read_csv(out/'match_sensitivity.csv')
    for horizon in [500,1000]:
        p=pd.read_parquet(out/f'pairs_{horizon}ms.parquet');w=pd.read_parquet(out/f'windows_{horizon}ms.parquet')
        assert not w.split.eq('test').any();assert not p.duplicated(['i','j']).any();assert (p.i<p.j).all()
        assert (w.log_sha.to_numpy()[p.i]!=w.log_sha.to_numpy()[p.j]).all()
        assert (w.hardware_uuid.to_numpy()[p.i]==w.hardware_uuid.to_numpy()[p.j]).all()
        assert (w.nav_state.to_numpy()[p.i]==w.nav_state.to_numpy()[p.j]).all()
        f=flights.set_index('flight_id')
        for fid,g in w.groupby('flight_id'):
            assert g.history_start_us.min()>=f.loc[fid,'start_us'];assert g.pair_support_end_us.max()<=f.loc[fid,'end_us']
            for a,b in prov[fid]['exclude_intervals_us']:
                assert not ((g.history_start_us<=b)&(g.pair_support_end_us>=a)).any()
        mask=p.primary&p[[c for c in p if c.startswith('contrast_')]].any(axis=1)
        s=summaries.query('horizon_ms==@horizon and caliper=="primary" and contrast_scale==1 and channel=="any" and group=="all"').iloc[0]
        assert int(mask.sum())==s.raw_pairs
        assert len(disjoint_pairs(p[mask],w))==s.nonoverlap_pairs
        selected=pd.read_csv(out/f'selected_{horizon}ms.csv');sel=selected[selected.channel=='any'].pair_index
        assert list(sel)==disjoint_pairs(p[mask],w)
    coverage=pd.read_csv(out/'coverage.csv');assert coverage.flights.sum()==161
    event=pd.read_csv(out/'event_counts_per_flight.csv')
    totals=event.groupby(['channel','scale']).events.sum().unstack()
    assert (totals[.5]>=totals[1]).all() and (totals[1]>=totals[2]).all()

def test_candidate_radius_contains_all_caliper_pairs():
    # Necessary mean-coordinate bound for exhaustive radius pruning, no KNN cap.
    rng=np.random.default_rng(6)
    for _ in range(25):
        a=Rotation.random(26,random_state=rng)
        rotvec=rng.normal(size=(26,3))*.1
        b=a*Rotation.from_rotvec(rotvec)
        angle=(a.inv()*b).magnitude()
        threshold=np.sqrt(np.mean(angle**2))
        assert np.max(np.abs(a.as_matrix().mean(0)-b.as_matrix().mean(0)))<=threshold+1e-12

def test_native_history_only_columns():
    from audit import META,V,Q,W,U
    assert not any(c.startswith('target_') for c in META+V+Q+W+U)
    assert not any('position_' in c for c in V+Q+W+U)
    assert not any('phase' in c for c in V+Q+W+U)

def test_initial_mismatch_cannot_hide_in_history_average():
    from audit import caliper_mask
    x=np.zeros((1,26,15));x[:,:,3]=1;y=x.copy();y[:,-1,0]=1.0
    error=history_errors(x,y)
    assert error[0,0]<.5 and error[0,8]>.5
    assert not caliper_mask(error,dict(v=.5,R_deg=10,w=.5,f=.3,u=.03))[0]

def test_derived_pair_tables_and_independent_contrasts():
    out=HERE/'results'
    if not (out/'selected_pairs.csv.gz').exists():pytest.skip('postprocessing needed')
    selected=pd.read_csv(out/'selected_pairs.csv.gz')
    summary=pd.read_csv(out/'match_sensitivity.csv').query('caliper=="primary" and contrast_scale==1 and group=="all"')
    for row in summary.itertuples():
        q=selected[(selected.horizon_ms==row.horizon_ms)&(selected.channel==row.channel)]
        assert len(q)==row.nonoverlap_pairs
        assert len(set(q.flight_a)|set(q.flight_b))==row.nonoverlap_flights
        for flight in set(q.flight_a)|set(q.flight_b):
            spans=[(r.start_a,r.end_a) for r in q[q.flight_a==flight].itertuples()]
            spans += [(r.start_b,r.end_b) for r in q[q.flight_b==flight].itertuples()]
            spans.sort()
            assert all(spans[i][1]<spans[i+1][0] for i in range(len(spans)-1))
        if row.channel!='any':
            channels=['motor','left_elevon','right_elevon','rudder']
            for c,limit in zip(channels,[.02,.04,.04,.02]):
                if c!=row.channel:assert (q['future_gap_'+c]<=limit+1e-12).all()
    m=json.loads((out/'run_manifest.json').read_text())
    assert m['code_sha256']==sha(HERE/'audit.py')
    assert m['config_sha256']==sha(HERE/'config.json')
    assert m['test_artifacts_opened'] is False and m['targets_read'] is False
    v=json.loads((out/'scalar_verification.json').read_text())
    assert [x['scalar_recomputed_pairs'] for x in v['results']]==[100,100]
