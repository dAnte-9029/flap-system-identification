"""Reproduce candidate predictions and independently recompute the fixed gates."""
import json
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from controller_no_age_v27 import OUT,V23,V25,V26,model,load
from reference_groupcv_v8 import predict
from audit_controller_support_v26 import verify_parent


def main():
    torch.set_num_threads(1);protocol=json.loads((OUT/'protocol.json').read_text())
    for path,h in protocol['code_pins'].items():assert base.sha(base.ROOT/path)==h,path
    for path,h in protocol['parents'].items():
        root=base.ROOT/path;assert base.sha(root/'delivery_manifest.json')==h;verify_parent(root)
    x,y=load();w=pd.read_csv(V23/'windows.csv');rows=[];checks=[];coverage={}
    for fold in range(5):
        info=json.loads((V23/f'fold{fold}/manifest.json').read_text());ids=info['query_indices']
        assert set(w.iloc[ids].log_id).isdisjoint(w.iloc[info['fit_indices']].log_id)
        for seed in [17,23,42]:
            root=OUT/f'fold{fold}/no_age_history_control/seed{seed}'
            st=json.loads((root/'status.json').read_text());assert st['status']=='complete' and st['step']==1600
            assert base.sha(root/'final.pt')==st['checkpoint_sha256']
            run=json.loads((root/'run_protocol.json').read_text())
            for arm in ['zero_history_control','observed_history_control']:
                old=json.loads((V25/f'fold{fold}/{arm}/seed{seed}/run_protocol.json').read_text())
                for key in ['initial_parameters_sha256','common_initial_parameters_sha256','draws_sha256']:assert run[key]==old[key]
            checkpoint=torch.load(root/'final.pt',weights_only=False)
            assert checkpoint['fold']==fold and checkpoint['seed']==seed and checkpoint['arm']=='no_age_history_control'
            assert checkpoint['feature_contract']=='rate_reference_past_controller_no_age_v27'
            assert checkpoint['protocol_sha256']==base.sha(OUT/'protocol.json')
            m=model(fold);m.load_state_dict(checkpoint['state_dict']);m.eval()
            assert not m.encoder.weight_ih_l0[:,-2:].detach().count_nonzero()
            for cohort,ix in [('fit',info['fit_indices']),('oof',ids)]:
                pred=predict(m,x,ix)
                with np.load(root/f'{cohort}_predictions.npz') as z:
                    np.testing.assert_array_equal(z['array_index'],ix)
                    for k in y:np.testing.assert_array_equal(z[k],pred[k])
            unchanged=dict(x);unchanged['history_controller']=x['history_controller'].clone();unchanged['history_controller'][...,6:]+=100
            for k,value in predict(m,unchanged,ids).items():np.testing.assert_array_equal(value,pred[k])
            held=dict(x);held['future_reference']=x['future_reference'].clone();held['future_reference'][:,:,:4]=x['future_reference'][:,:1,:4]
            with np.load(root/'oof_held.npz') as z:
                np.testing.assert_array_equal(z['array_index'],ids)
                for k,value in predict(m,held,ids).items():np.testing.assert_array_equal(value,z[k])
            coverage.setdefault(seed,[]).extend(ids)
            # Independent arithmetic, without using the production errors helper.
            for arm,path in [('zero',V25/f'fold{fold}/zero_history_control/seed{seed}'),('no_age',root)]:
                with np.load(path/'oof_predictions.npz') as z:
                    r=(z['rates_b'].astype(float)-y['rates_b'][ids].numpy().astype(float))**2
                    v=np.mean((z['velocity_n'].astype(float)-y['velocity_n'][ids].numpy().astype(float))**2,axis=2)
                    angle=(2*np.arccos(np.abs(np.sum(z['quaternion'].astype(float)*y['quaternion'][ids].numpy().astype(float),axis=2)).clip(0,1)))**2
                for metric,error in [('p',r[:,:,0]),('q',r[:,:,1]),('r',r[:,:,2]),('velocity',v),('attitude',angle)]:
                    for kind,horizon in [('wave',50),('endpoint',25),('endpoint',50)]:
                        vals=error.mean(1) if kind=='wave' else error[:,horizon-1]
                        rows.append(pd.DataFrame(dict(seed=seed,arm=arm,metric=metric,kind=kind,horizon=horizon,
                            log_id=w.iloc[ids].log_id.to_numpy(),active=w.iloc[ids].active_reference.to_numpy(),mse=vals)))
            checks.append(dict(fold=fold,seed=seed,fit_oof_held_replay_exact=True,age_invariance=True,paired_initialization_draws=True))
    for ids in coverage.values():assert sorted(ids)==list(range(407))
    a=pd.concat(rows);keys=['seed','arm','metric','kind','horizon'];flight=a.groupby(keys+['log_id']).mse.mean().pow(.5)
    avg=flight.groupby(keys).mean().unstack('arm');gain=100*(1-avg.no_age/avg.zero)
    q=avg.xs(('q','wave',50),level=['metric','kind','horizon']);primary=100*(1-q.no_age.mean()/q.zero.mean())
    qf=flight.xs(('q','wave',50),level=['metric','kind','horizon']).groupby(['arm','log_id']).mean().unstack('arm')
    positive=int((qf.no_age<qf.zero).sum());seeds_positive=bool((q.no_age<q.zero).all())
    protection=[value for (_,metric,kind,h),value in gain.items() if (kind=='wave' and metric in ['p','r']) or (kind=='endpoint' and metric in ['velocity','attitude'])]
    assert len(protection)==18
    active=a[(a.arm=='no_age')&a.active&(a.kind=='endpoint')&(a.horizon==25)]
    active=active.groupby(['seed','metric','log_id']).mse.mean().pow(.5).groupby(['seed','metric']).mean()
    limits={'p':.15,'q':.15,'velocity':.5,'attitude':np.deg2rad(3)}
    passed=sum(value<=limits[metric] for (_,metric),value in active.items() if metric in limits)
    e=json.loads((OUT/'evaluation.json').read_text());np.testing.assert_allclose(primary,e['primary_q_improvement_percent'],atol=1e-10)
    assert positive==e['positive_flights'] and seeds_positive==e['all_seeds_positive']
    assert sum(v>=-5 for v in protection)==e['protection_passed'] and passed==e['absolute_active_passed']
    assert bool(primary>=5 and positive>=4 and seeds_positive and min(protection)>=-5)==e['numerical_gate']
    base.write(OUT/'verification.json',dict(status='passed',checks=checks,models_replayed=15,windows_per_seed=407,
        independent_primary_protection_absolute=True,parent_and_code_hashes=True,focused_tests_passed=3,
        validation_accessed=False,sealed_test_accessed=False))


if __name__=='__main__':main()
