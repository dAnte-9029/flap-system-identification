"""Independently replay v35 checkpoints and recompute frozen v34 gates."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from actuator_training_v30 import load, verify_parent, model_from
from train_joint_increment_v34 import OUT, DATA, CONTROL, PREP, verify_frozen as verify_protocol
from reference_groupcv_v8 import predict
from prepare_actuator_inputs_v29 import fingerprint
from system_identification.models.actuator_command_gru import ActuatorCommandGRU


def read(path): return json.loads(path.read_text())


def main():
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    verify_protocol()
    for root in [DATA, CONTROL]: verify_parent(root)
    assert read(OUT / 'launch/status.json')['status'] == 'training_complete_evaluation_pending'
    x, y = load(); w = pd.read_csv(DATA / 'windows.csv')
    metric_rows = []; checks = []; seen = {s: [] for s in [17,23,42]}
    for run in read(PREP / 'training_plan.json')['runs']:
        fold, seed = run['fold'], run['seed']; root = OUT / f'fold{fold}/joint_increment/seed{seed}'
        info = read(DATA / f'fold{fold}/manifest.json'); ids = info['query_indices']; seen[seed].extend(ids)
        status = read(root / 'status.json')
        assert status['status'] == 'complete' and status['step'] == 1600
        assert base.sha(root / 'final.pt') == status['checkpoint_sha256']
        rp = read(root / 'run_protocol.json')
        for k,v in run.items(): assert rp[k] == v, k
        assert rp['execution_protocol_sha256'] == base.sha(OUT / 'protocol.json')
        model = model_from(root / 'final.pt')
        ck = torch.load(root / 'final.pt', map_location='cpu', weights_only=False)
        assert (ck['fold'],ck['seed'],ck['steps']) == (fold,seed,1600)
        assert ck['protocol_sha256'] == base.sha(PREP / 'protocol.json')
        assert ck['local_weight'] == rp['weight'] == 1.0
        assert rp['preparation_protocol_sha256'] == base.sha(PREP / 'protocol.json')
        with np.load(DATA / f'fold{fold}/normalization.npz') as z:
            stats = {k:z[k] for k in z.files}
            for k,v in stats.items(): np.testing.assert_array_equal(getattr(model,k).numpy(), v.astype(np.float32))
        torch.manual_seed(seed); init = ActuatorCommandGRU(stats)
        assert fingerprint(init) == run['initial_parameter_sha256']
        assert sum(p.numel() for p in model.parameters()) == 40070
        assert model.reference_mean[4] == 0 and model.reference_scale[4] == 1
        held = dict(x); held['future_command'] = x['future_command'][:, :1].expand(-1,50,-1)
        for name, data, indices in [('fit',x,info['fit_indices']),('oof',x,ids),('oof_held',held,ids)]:
            pred = predict(model, data, indices)
            path = root / (name+'.npz' if name=='oof_held' else name+'_predictions.npz')
            with np.load(path) as z:
                np.testing.assert_array_equal(z['array_index'],indices)
                for k,v in pred.items(): np.testing.assert_array_equal(v,z[k])
        changed = dict(x); changed['future_command'] = x['future_command'].clone()
        changed['future_command'][:,10:] += .1
        a = predict(model,x,ids[:2]); b = predict(model,changed,ids[:2])
        for k in a: np.testing.assert_array_equal(a[k][:,:10],b[k][:,:10])
        # Independent OOF metric computation, without evaluate_actuator_v30.errors.
        truth = {k:v[ids].numpy().astype(float) for k,v in y.items()}
        for arm, directory in [('baseline',CONTROL/f'fold{fold}/seed{seed}'),('joint',root)]:
            with np.load(directory/'oof_predictions.npz') as z:
                np.testing.assert_array_equal(z['array_index'],ids)
                rates = (z['rates_b'].astype(float)-truth['rates_b'])**2
                velocity = ((z['velocity_n'].astype(float)-truth['velocity_n'])**2).mean(-1)
                angle = 2*np.arccos(np.clip(np.abs((z['quaternion'].astype(float)*truth['quaternion']).sum(-1)),0,1))
            quantities = dict(p=rates[:,:,0],q=rates[:,:,1],r=rates[:,:,2],velocity=velocity,attitude=angle**2)
            active = w.iloc[ids].active_reference.to_numpy(bool)
            for key, squared in quantities.items():
                for horizon in [1,5,25,50]:
                    for kind, value in [('wave',squared[:,:horizon].mean()),('endpoint',squared[:,horizon-1].mean())]:
                        metric_rows.append(dict(seed=seed,fold=fold,arm=arm,metric=key,horizon=horizon,kind=kind,rmse=float(np.sqrt(value))))
                if key != 'r':
                    metric_rows.append(dict(seed=seed,fold=fold,arm=arm,metric=key,horizon=25,kind='active',
                                            rmse=float(np.sqrt(squared[active,24].mean()))))
        checks.append(dict(fold=fold,seed=seed,all_predictions_exactly_replayed=True,normalization_buffers_exact=True,
                           initial_parameters_match=True,prefix_causality=True,parameters=40070))
    for values in seen.values(): np.testing.assert_array_equal(sorted(values),np.arange(407))
    independent = pd.DataFrame(metric_rows)
    comp = independent[independent.kind!='active'].groupby(['seed','metric','horizon','kind','arm']).rmse.mean().unstack()
    actual = pd.read_csv(OUT/'comparison.csv'); actual = actual[actual.cohort=='oof'].set_index(['seed','metric','horizon','kind'])
    np.testing.assert_allclose(comp[['baseline','joint']],actual.loc[comp.index,['baseline','joint']],rtol=1e-12,atol=1e-12)
    q = comp.reset_index(); q = q[(q.metric=='q')&(q.horizon==50)&(q.kind=='wave')]
    improvement = float(100*(1-q.joint.mean()/q.baseline.mean()))
    f = independent[(independent.metric=='q')&(independent.horizon==50)&(independent.kind=='wave')]
    flights = f.groupby(['fold','arm']).rmse.mean().unstack(); positive = int((flights.joint<flights.baseline).sum())
    c = comp.reset_index()
    protect = c[((c.kind=='wave')&(c.horizon==50)&c.metric.isin(['p','r']))|
                ((c.kind=='endpoint')&c.horizon.isin([25,50])&c.metric.isin(['velocity','attitude']))].copy()
    protect['improvement_percent'] = 100*(1-protect.joint/protect.baseline)
    protection_count = int((protect.improvement_percent>=-5).sum())
    active = independent[(independent.kind=='active')&(independent.arm=='joint')].groupby(['seed','metric']).rmse.mean()
    recorded = pd.read_csv(OUT/'absolute_active_gate.csv').set_index(['seed','metric'])
    np.testing.assert_allclose(active,recorded.loc[active.index,'rmse'],rtol=1e-12,atol=1e-12)
    limits = {'p':.15,'q':.15,'velocity':.5,'attitude':np.deg2rad(3)}
    absolute_count = sum(v<=limits[key] for (seed,key),v in active.items())
    e = read(OUT/'evaluation.json')
    np.testing.assert_allclose(improvement,e['primary_q_wave_improvement_percent'],rtol=0,atol=1e-10)
    assert positive == e['positive_flights'] and protection_count == e['protection_passed']
    assert len(protect) == e['protection_total'] == 18 and len(active) == e['absolute_active_total'] == 12
    assert absolute_count == e['absolute_active_passed']
    allseeds = bool((q.joint<q.baseline).all())
    assert allseeds == e['all_seeds_q_positive']
    assert (improvement>=5 and allseeds and positive>=4 and protection_count==18) == e['descriptive_gate']
    independent.to_csv(OUT/'independent_oof_metrics.csv',index=False)
    flights['improvement_percent'] = 100*(1-flights.joint/flights.baseline)
    flights.to_csv(OUT/'q_by_flight.csv')
    base.write(OUT/'verification.json',dict(status='passed',checks=checks,unique407_oof_per_seed=True,
        independent_all_oof_metrics_and_gates_match=True,parent_hashes_verified=True,
        validation_accessed=False,sealed_test_accessed=False,reserved_accessed=False))
    verify_protocol()
    print('All 45 saved prediction groups replayed exactly; independent OOF metrics and gates passed.',flush=True)


if __name__ == '__main__': main()
