"""Freeze, prepare causal warmup, and execute bounded reference-conditioned replay."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from pyulog import ULog
import train_control_windows_v2 as base
from audit_controller_support_v26 import verify_parent
from actuator_training_v30 import model_from
from run_c9af_replay_v38 import TOPICS, indices, euler, initialize
from coupled_replay_v40 import VARIANTS, rollout, teacher
from system_identification.control.px4_c9af import C9Controller, FIRMWARE

OUT=base.ROOT/'docs/analysis/results/control_windows_v40'
DATA=OUT.parent/'control_windows_v29';V37=OUT.parent/'control_windows_v37'
PARENTS=[DATA,*(OUT.parent/f'control_windows_v{v}' for v in [30,35,38,39])]


def read(p):return json.loads(p.read_text())


def json_scalar(value):
    if isinstance(value,np.generic):return value.item()
    raise TypeError(f'Unsupported JSON value {type(value).__name__}')


def prepare_flight(entry):
    assert base.sha(entry['source']['path'])==entry['source']['sha256']
    u=ULog(entry['source']['path'],message_name_filter_list=TOPICS)
    assert u.msg_info_dict['ver_sw']==FIRMWARE and not u.changed_parameters
    ds={d.name:d.data for d in u.data_list if d.multi_id==0}
    p={k:v for k,v in u.initial_parameters.items() if k.startswith(('FW_','CA_','VT_','TRIM_','PWM_MAIN_','FLAP_'))}
    assert all(p[f'FW_{a}R_D']==0 for a in 'RPY')
    w=pd.read_csv(DATA/'windows.csv');w=w[w.log_id==entry['log_id']]
    assert base.sha(base.ROOT/entry['cache'])==entry['cache_sha256']
    with np.load(base.ROOT/entry['cache']) as z:times=z['timestamp_us']
    att=euler(ds['vehicle_attitude'],'q');ref=euler(ds['vehicle_attitude_setpoint'],'q_d')
    rate=np.column_stack([ds['vehicle_angular_velocity'][f'xyz[{j}]'] for j in range(3)])
    snapshots=[];tapes=[];airs=[];manuals=[];origins=[];provenance=[]
    for row in w.itertuples():
        origin=int(times[row.origin_index]);query=origin+np.arange(400)*2500
        ri=indices(ds['vehicle_attitude_setpoint'],query);mi=indices(ds['manual_control_setpoint'],query)
        md=ds['vehicle_control_mode'];mdi=indices(md,query)
        vd=ds['vehicle_status'];vdi=indices(vd,query)
        ld=ds['vehicle_land_detected'];ldi=indices(ld,query)
        assert md['flag_control_attitude_enabled'][mdi].all() and md['flag_control_rates_enabled'][mdi].all()
        assert not vd['is_vtol'][vdi].any() and not ld['landed'][ldi].any()
        manual=bool(md['flag_control_manual_enabled'][mdi[0]])
        assert (md['flag_control_manual_enabled'][mdi]==manual).all()
        tape=np.column_stack([ref[ri],ds['vehicle_attitude_setpoint']['thrust_body[0]'][ri],ds['manual_control_setpoint']['yaw'][mi]])
        assert np.isfinite(tape).all()
        ai=indices(ds['airspeed_validated'],np.array([origin]))[0]
        cas=float(ds['airspeed_validated']['calibrated_airspeed_m_s'][ai])
        age=(origin-int(ds['airspeed_validated']['timestamp'][ai]))*1e-6
        # Same source stale/invalid fallback, then fixed fresh synthetic CAS proxy.
        air=cas if np.isfinite(cas) and age<1 else p['FW_AIRSPD_TRIM']
        airs.append(air);manuals.append(manual);tapes.append(tape);origins.append(origin)
        provenance.append(dict(array_index=int(row.array_index),origin_us=origin,air_source_index=int(ai),
            air_publication_us=int(ds['airspeed_validated']['timestamp'][ai]),air_age_s=age,
            air_fallback=bool(not np.isfinite(cas) or age>=1),
            attitude_reference_indices=ri.tolist(),manual_reference_indices=mi.tolist()))
        saved={}
        for name,tick,feedback in VARIANTS:
            c=C9Controller(p);start=origin-2000000
            initialize(c,ds,start,-1 if name=='warm2_minus' else 1)
            ticks=np.arange(start,origin,tick);ix={key:indices(ds[key],ticks) for key in TOPICS if key in ds}
            prev=-1
            for n,t in enumerate(ticks):
                mm=ix['vehicle_control_mode'][n];ss=ix['vehicle_status'][n]
                assert md['flag_control_rates_enabled'][mm] and md['flag_control_attitude_enabled'][mm] and not vd['is_vtol'][ss]
                av=ix['airspeed_validated'][n];aa=float(ds['airspeed_validated']['calibrated_airspeed_m_s'][av]);ag=(t-int(ds['airspeed_validated']['timestamp'][av]))*1e-6
                at=ix['vehicle_attitude'][n];rr=ix['vehicle_attitude_setpoint'][n];man=ix['manual_control_setpoint'][n]
                if at!=prev:
                    c.attitude(*att[at],*ref[rr],aa,ag,manual=bool(md['flag_control_manual_enabled'][mm]),yaw_stick=float(ds['manual_control_setpoint']['yaw'][man]));prev=at
                landed=bool(ld['landed'][ix['vehicle_land_detected'][n]])
                c.rate(rate[ix['vehicle_angular_velocity'][n]],c.state.last_rate_sp,[0.,0.,0.],aa,ag,
                       float(ds['vehicle_attitude_setpoint']['thrust_body[0]'][rr]),tick*1e-6,
                       landed=landed,reset_integral=landed,feedback=feedback)
            # Feedback comparisons may yield numpy.bool_; preserve scalar values
            # and strict nonfinite rejection rather than dropping state fields.
            saved[name]=json.loads(json.dumps(c.snapshot(),default=json_scalar,allow_nan=False))
        snapshots.append(dict(array_index=int(row.array_index),states=saved))
    dest=OUT/Path(entry['source']['path']).stem;dest.mkdir()
    np.savez_compressed(dest/'conditions.npz',array_index=w.array_index.to_numpy(),origin_us=origins,tape=tapes,air=airs,manual=manuals)
    base.write(dest/'snapshots.json',snapshots);base.write(dest/'provenance.json',provenance)
    return dict(log_id=entry['log_id'],path=str(dest.relative_to(OUT)),windows=len(w),
        artifacts={f:base.sha(dest/f) for f in ['conditions.npz','snapshots.json','provenance.json']})


def worker(fold):
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    protocol=read(OUT/'protocol.json')
    for path,digest in protocol['code'].items():assert base.sha(base.ROOT/path)==digest
    info=read(DATA/f'fold{fold}/manifest.json');ids=info['query_indices'];fit=info['fit_indices']
    with np.load(DATA/'windows.npz') as z:
        history={k[6:]:torch.tensor(z[k][ids]) for k in z.files if k.startswith('input_') and k not in ['input_future_command','input_dt_s']}
        tq=np.concatenate([z['input_initial_quaternion'][ids,None],z['label_quaternion'][ids,:49]],1)
        tr=np.concatenate([z['input_initial_rates_b'][ids,None],z['label_rates_b'][ids,:49]],1)
        commands=z['input_future_command'][fit];states=z['input_history_state'][fit]
        support=dict(command_min=commands.min((0,1)),command_max=commands.max((0,1)),
                     state_min=states.min((0,1)),state_max=states.max((0,1)))
    entry=next(e for e in read(OUT/'conditions_manifest.json')['flights'] if set(np.load(OUT/e['path']/'conditions.npz')['array_index'])==set(ids))
    src=OUT/entry['path']
    for name,digest in entry['artifacts'].items():assert base.sha(src/name)==digest
    with np.load(src/'conditions.npz') as z:
        order=[list(z['array_index']).index(i) for i in ids]
        tape=z['tape'][order];air=z['air'][order];manual=z['manual'][order]
    snaps={x['array_index']:x['states'] for x in read(src/'snapshots.json')}
    dest=OUT/f'fold{fold}';dest.mkdir()
    base.write(dest/'support.json',dict(fit_indices=fit,query_indices=ids,definition='Per-channel min/max on fit future commands and fit history state14; marginal box only, not joint support or confidence.',**{k:v.tolist() for k,v in support.items()}))
    count=0
    for variant in VARIANTS:
        name=variant[0];snap=[snaps[i][name] for i in ids]
        teach=teacher(tq,tr,snap,tape,air,manual,variant)
        np.savez_compressed(dest/(name+'_teacher.npz'),array_index=ids,**teach)
        for stage in [30,35]:
            for seed in [17,23,42]:
                spec=next(r for r in protocol['models'] if (r['stage'],r['fold'],r['seed'])==(stage,fold,seed))
                assert base.sha(base.ROOT/spec['checkpoint'])==spec['sha256']
                model=model_from(base.ROOT/spec['checkpoint'])
                # Every OOF model keeps its own fitted normalization unchanged.
                with np.load(DATA/f'fold{fold}/normalization.npz') as z:
                    for k in z.files:np.testing.assert_array_equal(getattr(model,k).numpy(),z[k].astype(np.float32))
                result=rollout(model,history,snap,tape,air,manual,variant,support)
                assert all(np.isfinite(v).all() for v in result.values())
                np.savez_compressed(dest/f'{name}_v{stage}_seed{seed}.npz',array_index=ids,**result)
                count+=1
                base.write(dest/'status.json',dict(status='running',completed=count,total=24))
        print(f'fold{fold} {name}: teacher +6 model runs complete',flush=True)
    base.write(dest/'status.json',dict(status='complete',completed=count,total=24,all_windows=len(ids)))
    return fold


def main():
    for parent in PARENTS:verify_parent(parent)
    names=['scripts/coupled_replay_v40.py','scripts/run_coupled_replay_v40.py','tests/test_coupled_replay_v40.py']
    models=[]
    for r in read(OUT.parent/'control_windows_v39/protocol.json')['runs']:
        models.append(dict(stage=r['stage'],fold=r['fold'],seed=r['seed'],checkpoint=r['checkpoint'],sha256=r['checkpoint_sha256']))
    OUT.mkdir(exist_ok=False)
    protocol=dict(stage='v40 bounded reference-conditioned controller-model coupling',status='frozen_before_preparation_and_execution',
        code={n:base.sha(base.ROOT/n) for n in names},parents={str(p.relative_to(base.ROOT)):base.sha(p/'delivery_manifest.json') for p in PARENTS},
        models=models,variants=[dict(name=n,tick_us=t,feedback=f) for n,t,f in VARIANTS],
        data='All407 originaltrain parents, all30 OOF checkpoints; no outcome selection, refit or additional clamps.',
        initialization='Causal2s raw-state warmup ending strictly before origin, HPF all-plus/all-minus probes; original modelH26 initialization.',
        clock='Model20ms; controller2.5ms/attitude10ms or20ms alternative. Hold model state for20ms, consume first generated action for native model step; record all intermediate actions. This is a50Hz plant-input hold approximation, not recovered firmware scheduling.',
        inputs='400x2.5ms causal publication+sample held logged attitude/thrust/manualyaw condition tape. Fixed origin manual/mode, all scored airborne attitude/rate/nonVTOL flags audited. No future plant measurements or logged actuator commands enter model rollout.',
        proxies='Controller uses fixed origin valid CAS (source trim fallback if stale/invalid) as fresh synthetic airspeed. Model keeps original frequency/phase/air proxies. D=0 verified, controller acceleration0, no future reset state.',
        teacher='Same coupling schedule, initial controller state, fixedCAS and condition tape, held measured20ms state, no model; diagnostic only. v38 retained as different-schedule reference.',
        support='Per-fold fit-only command min/max and observed history state14 min/max, evaluated on generated/current quantities. Marginal coverage only, not calibrated confidence or joint support.',
        evaluation='Full407; equal-flight then equal-seed. State full1s wave and.5s/1s endpoint; generated action/teacher action error, microcommand mismatch, saturation, support and all sensitivity variants. Failures retained, no skipping/restarts to hide failure.',
        gates=read(DATA/'protocol.json')['descriptive_gate'],
        gate_application='Apply frozen relative/protection/absolute definitions to v35 versus v30 at identical variant. Additional action diagnostic: per-seed, each four-channel action RMSE may regress at most5% vs v30; each passed cell reported, not physical fidelity certification. No promotion even if all descriptive gates pass.',
        deployment=False,training=False,validation_accessed=False,sealed_test_accessed=False,reserved_accessed=False)
    base.write(OUT/'protocol.json',protocol)
    base.write(OUT/'dataset_manifest.json',dict(authority=str((DATA/'dataset_manifest.json').relative_to(base.ROOT)),
        authority_sha256=base.sha(DATA/'dataset_manifest.json'),contract=read(DATA/'dataset_manifest.json'),
        partitions=['original_train_only'],teacher_future_state='Only separate teacher diagnostic and scoring'))
    base.write(OUT/'status.json',dict(status='preparing_causal_warmup'))
    entries=read(V37/'dataset_manifest.json')['sources']
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as pool:
        flights=list(pool.map(prepare_flight,entries))
    base.write(OUT/'conditions_manifest.json',dict(flights=flights,all407=True))
    base.write(OUT/'status.json',dict(status='running_coupled_replay',completed_folds=[]))
    completed=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as pool:
        for fold in pool.map(worker,range(5)):
            completed.append(fold);base.write(OUT/'status.json',dict(status='running_coupled_replay',completed_folds=completed))
    for parent in PARENTS:verify_parent(parent)
    base.write(OUT/'status.json',dict(status='replay_complete_evaluation_pending',completed_folds=completed))


if __name__=='__main__':main()
