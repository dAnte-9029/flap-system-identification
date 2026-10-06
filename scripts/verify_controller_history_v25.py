"""Replay every saved OOF prediction and independently recompute frozen gates."""
import json
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from controller_history_training_v25 import OUT,V23,V24,V8,ARMS,load,model
from reference_groupcv_v8 import predict


def main():
    torch.set_num_threads(1)
    protocol=json.loads((OUT/'protocol.json').read_text())
    for path,digest in protocol['code_pins'].items():assert base.sha(base.ROOT/path)==digest,path
    for root in [V23,V24]:
        assert base.sha(root/'delivery_manifest.json')==protocol['parent_delivery_sha256'][str(root.relative_to(base.ROOT))]
        manifest=json.loads((root/'delivery_manifest.json').read_text())
        for path,digest in manifest['artifacts'].items():assert base.sha(root/path)==digest,path
        for path,digest in manifest['code'].items():assert base.sha(base.ROOT/path)==digest,path
    x,y=load();windows=pd.read_csv(V23/'windows.csv');records=[];covered={};replays=[]
    assert len(windows)==407 and windows.array_index.tolist()==list(range(407))
    for fold in range(5):
        info=json.loads((V23/f'fold{fold}/manifest.json').read_text())
        fit,ids=info['fit_indices'],info['query_indices']
        assert set(fit).isdisjoint(ids) and set(fit)|set(ids)==set(range(407))
        assert set(windows.iloc[fit].log_id).isdisjoint(windows.iloc[ids].log_id)
        for seed in [17,23,42]:
            paired=[]
            for arm in ARMS:
                root=OUT/f'fold{fold}/{arm}/seed{seed}';status=json.loads((root/'status.json').read_text())
                run=json.loads((root/'run_protocol.json').read_text());paired.append(run)
                assert status['status']=='complete' and status['step']==1600
                assert status['checkpoint_sha256']==base.sha(root/'final.pt')
                assert run['parent_protocol_sha256']==base.sha(OUT/'protocol.json')
                old=V8/f'fold{fold}/hold_{seed}'
                assert run['draws_sha256']==base.sha(old/'sample_draws.npz')
                assert run['common_initial_parameters_sha256']==json.loads((old/'run_protocol.json').read_text())['initial_parameter_sha256']
                m=model(fold,arm);checkpoint=torch.load(root/'final.pt',weights_only=False)
                assert checkpoint['arm']==arm and checkpoint['seed']==seed and checkpoint['fold']==fold
                assert checkpoint['protocol_sha256']==base.sha(OUT/'protocol.json')
                m.load_state_dict(checkpoint['state_dict']);m.eval()
                if arm==ARMS[0]:assert not m.encoder.weight_ih_l0[:,25:].detach().count_nonzero()
                p=predict(m,x,ids)
                with np.load(root/'oof_predictions.npz') as z:
                    np.testing.assert_array_equal(z['array_index'],ids)
                    for key in y:np.testing.assert_array_equal(z[key],p[key])
                if arm==ARMS[0]:
                    for cohort in ['fit','oof']:
                        with np.load(root/f'{cohort}_predictions.npz') as z,np.load(V23/f'fold{fold}/seed{seed}/{cohort}_predictions.npz') as old_predictions:
                            for key in z.files:np.testing.assert_array_equal(z[key],old_predictions[key])
                held=dict(x);held['future_reference']=x['future_reference'].clone()
                held['future_reference'][:,:,:4]=x['future_reference'][:,:1,:4]
                hp=predict(m,held,ids)
                with np.load(root/'oof_held.npz') as z:
                    np.testing.assert_array_equal(z['array_index'],ids)
                    for key in y:np.testing.assert_array_equal(z[key],hp[key])
                covered.setdefault((seed,arm),[]).extend(ids)
                true={key:value[ids].numpy().astype(float) for key,value in y.items()}
                rate_error=(p['rates_b'].astype(float)-true['rates_b'])**2
                velocity_error=np.mean((p['velocity_n'].astype(float)-true['velocity_n'])**2,axis=2)
                angle_error=(2*np.arccos(np.clip(np.abs(np.sum(p['quaternion'].astype(float)*true['quaternion'],axis=2)),0,1)))**2
                for metric,error in [('p',rate_error[:,:,0]),('q',rate_error[:,:,1]),('r',rate_error[:,:,2]),('velocity',velocity_error),('attitude',angle_error)]:
                    for kind,h in [('wave',50),('endpoint',25),('endpoint',50)]:
                        mse=error.mean(axis=1) if kind=='wave' else error[:,h-1]
                        records.append(pd.DataFrame(dict(seed=seed,arm=arm,log_id=windows.iloc[ids].log_id.to_numpy(),
                            metric=metric,kind=kind,horizon=h,mse=mse,active=windows.iloc[ids].active_reference.to_numpy())))
                replays.append(dict(fold=fold,seed=seed,arm=arm,windows=len(ids),full_oof_exact=True,full_held_exact=True))
            for key in ['initial_parameters_sha256','common_initial_parameters_sha256','draws_sha256']:
                assert paired[0][key]==paired[1][key]
    for ids in covered.values():assert sorted(ids)==list(range(407))
    raw=pd.concat(records,ignore_index=True)
    keys=['seed','arm','metric','kind','horizon']
    flights=raw.groupby(keys+['log_id']).mse.mean().pow(.5)
    avg=flights.groupby(keys).mean().unstack('arm')
    gain=100*(1-avg[ARMS[1]]/avg[ARMS[0]])
    q=avg.xs(('q','wave',50),level=['metric','kind','horizon'])
    qgain=100*(1-q[ARMS[1]].mean()/q[ARMS[0]].mean())
    qf=flights.xs(('q','wave',50),level=['metric','kind','horizon']).groupby(['arm','log_id']).mean().unstack('arm')
    positive=int((qf[ARMS[1]]<qf[ARMS[0]]).sum())
    protection=[value for (_,metric,kind,h),value in gain.items() if
                (kind=='wave' and metric in ['p','r']) or (kind=='endpoint' and metric in ['velocity','attitude'])]
    assert len(protection)==18
    a=raw[(raw.arm==ARMS[1])&raw.active&(raw.kind=='endpoint')&(raw.horizon==25)]
    a=a.groupby(['seed','metric','log_id']).mse.mean().pow(.5).groupby(['seed','metric']).mean()
    limits={'p':.15,'q':.15,'velocity':.5,'attitude':np.deg2rad(3)}
    passed=sum(value<=limits[metric] for (_,metric),value in a.items() if metric in limits)
    result=json.loads((OUT/'evaluation.json').read_text())
    np.testing.assert_allclose(qgain,result['primary_q_improvement_percent'],atol=1e-10)
    assert positive==result['positive_flights']
    assert sum(value>=-5 for value in protection)==result['protection_passed']
    assert passed==result['absolute_active_passed'] and result['absolute_active_total']==12
    seed_positive=bool((q[ARMS[1]]<q[ARMS[0]]).all())
    assert seed_positive==result['all_seeds_positive']
    assert bool(qgain>=5 and seed_positive and positive>=4 and min(protection)>=-5)==result['numerical_gate']
    base.write(OUT/'verification.json',dict(status='passed',checkpoints_replayed=30,full_oof_replays=replays,
        paired_initialization_and_draws=True,each_seed_arm_exactly_407_unique_oof=True,
        zero_vs_v23_full_fit_oof_exact=True,zero_vs_v23_cohorts=30,zero_vs_v23_max_abs_difference=0.,
        independent_primary_protection_absolute_recomputed=True,parent_and_frozen_code_hashes=True,
        validation_accessed=False,sealed_test_accessed=False))


if __name__=='__main__':main()
