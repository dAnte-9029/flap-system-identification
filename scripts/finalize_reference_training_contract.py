"""Add omitted pilot-yaw reference and event-held mission metadata before training."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from pyulog import ULog
from audit_control_reference_semantics import past_values
from system_identification.data.reference_windows import extract_window

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/analysis/results/reference_dynamics_data_v1'
OUT=BASE/'training_contract'
CACHE=ROOT/'dataset/reference_dynamics_data_v1/contract'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')


def run():
    OUT.mkdir(parents=True,exist_ok=False);CACHE.mkdir(parents=True,exist_ok=False)
    parent=json.loads((BASE/'training_ready/manifest.json').read_text())
    assert sha(ROOT/parent['windows_path'])==parent['windows_sha256']
    base_manifest=json.loads((ROOT/parent['parent_manifest']).read_text())
    w=pd.read_csv(ROOT/parent['windows_path'])
    write(OUT/'protocol.json',dict(parent_sha256=sha(BASE/'training_ready/manifest.json'),script_sha256=sha(__file__),
        scope='same selected firmware/config cohort; augment pilot-yaw command reference; causal event-held mission subtype metadata',
        input_order=['roll_reference_rad','pitch_reference_rad','logged_yaw_sp_move_rate','thrust_body_x','manual_yaw_normalized'],
        current_mode_context='nav3/nav15 one-hot; not future observed mode',
        manual_yaw='Stabilized: last valid manual sample with publication/sample<=query and age<=0.3s; Mission: manual path inactive under selected flags, set0; do not pretend200ms logging is high-rate truth',
        mission_kind='last published current.type/current.valid held until next event; record age, not assume periodic publisher; type0 is position target, not proof of geometrically straight/level flight',
        selection='retain full original selected windows only if additional reference finite/fresh throughout history and future; no response metric selection',
        chain_boundary='archived source reveals manual yaw additive path; logged yaw_sp_move_rate alone is insufficient; matching firmware source and hardware still need verification before control claims',
        source_evidence_sha256=sha(ROOT/'docs/analysis/results/flight_chain_alignment/sources/src/modules/fw_att_control/FixedwingAttitudeControl.cpp'),
        sealed_test_accessed=False,training_started=False))
    rows=[];rejected=[];pins={};state=[];reference=[];actuator=[];extra_configs={}
    for path,items in w.groupby('cache'):
        log=items.log_id.iloc[0];src=base_manifest['sources'][log]
        assert sha(src['path'])==src['sha256']
        assert sha(ROOT/path)==base_manifest['cache_sha256'][path]
        with np.load(ROOT/path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        u=ULog(src['path'],message_name_filter_list=['manual_control_setpoint','position_setpoint_triplet'])
        ds={d.name:d.data for d in u.data_list if d.multi_id==0}
        manual,ma,msa=past_values(ds['manual_control_setpoint'],['yaw','valid'],a['timestamp_us'])
        stabilized=a['nav_state']==15
        manualgood=np.isfinite(manual).all(1)&(manual[:,1]==1)&(np.maximum(ma,msa)<=.3)
        yaw=np.where(stabilized,manual[:,0],0)
        a['references']=np.column_stack([a['references'],yaw]).astype(np.float32)
        a['manual_reference_age_s']=np.maximum(ma,msa).astype(np.float32)
        a['valid']=a['valid']&(~stabilized|manualgood)
        trip,ta,_=past_values(ds['position_setpoint_triplet'],['current.type','current.valid'],a['timestamp_us'])
        a['mission_type']=np.where((trip[:,1]==1)&np.isfinite(trip[:,0]),trip[:,0],-1)
        a['mission_type_age_s']=ta.astype(np.float32)
        extra={k:float(v) for k,v in u.initial_parameters.items() if k.startswith('TRIM_') or k in ['MAN_YR_MAX','SYS_AUTOSTART']}
        extra_configs[log]=extra
        dest=CACHE/Path(path).name;np.savez_compressed(dest,**a);pins[str(dest.relative_to(ROOT))]=sha(dest)
        retained=[]
        for row in items.to_dict('records'):
            i=int(row['origin_index'])
            if not a['valid'][i-26:i+51].all():
                rejected.append(dict(window_id=row['window_id'],reason='manual_reference_missing_invalid_or_stale'));continue
            inputs,labels=extract_window(a,i)
            assert inputs['future_reference'].shape==(50,5)
            row['cache']=str(dest.relative_to(ROOT));typ=a['mission_type'][i-26:i+51]
            row['kind']='stabilized' if row['mode']==15 else 'mission_position_target' if (typ==0).all() else 'mission_loiter_target' if (typ==2).all() else 'mission_transition_or_unknown'
            row['manual_yaw_range']=float(np.ptp(inputs['future_reference'][:,4]))
            row['active_reference']=bool(row['active_reference'] or row['manual_yaw_range']>=.03)
            rows.append(row);retained.append(i)
        if items.partition.iloc[0]=='train' and retained:
            hist=np.unique(np.concatenate([np.arange(i-25,i+1) for i in retained]));refs=np.unique(np.concatenate([np.arange(i-25,i+50) for i in retained]))
            state.append(a['state_features'][hist]);reference.append(a['references'][refs]);actuator.append(a['actuator_commands'][hist-1])
    # Fail rather than silently mixing an omitted trim/manual scaling difference.
    assert len({json.dumps(x,sort_keys=True) for x in extra_configs.values()})==1,'Extra control parameters differ; regroup required'
    write(OUT/'additional_control_parameters.json',extra_configs)
    f=pd.DataFrame(rows);f.to_csv(OUT/'windows.csv',index=False)
    pd.DataFrame(rejected,columns=['window_id','reason']).to_csv(OUT/'additional_rejections.csv',index=False)
    stats={}
    for name,items in [('state',state),('reference',reference),('past_actuator',actuator)]:
        x=np.concatenate(items).astype(float);assert np.isfinite(x).all()
        stats[name+'_mean']=x.mean(0);stats[name+'_scale']=x.std(0).clip(1e-6)
    stats['state_mean'][9:11]=0;stats['state_scale'][9:11]=1
    norm=CACHE/'normalization.npz';np.savez(norm,**stats);pins[str(norm.relative_to(ROOT))]=sha(norm)
    f.groupby(['partition','mode','kind','active_reference']).size().rename('windows').reset_index().to_csv(OUT/'summary.csv',index=False)
    for split in ['train','validation']:
        for mode in [3,15]:f[(f.partition==split)&(f['mode']==mode)].to_csv(OUT/f'{split}_mode{mode}.csv',index=False)
    write(OUT/'manifest.json',dict(dataset_id='reference_dynamics_data_v1_contract',group=parent['group'],
        parent_sha256=sha(BASE/'training_ready/manifest.json'),source_dataset_manifest_sha256=base_manifest['parent_manifest_sha256'],
        windows_path=str((OUT/'windows.csv').relative_to(ROOT)),windows_sha256=sha(OUT/'windows.csv'),cache_sha256=pins,
        normalization_path=str(norm.relative_to(ROOT)),normalization_partitions=['train'],
        input_reference_width=5,history_state_width=14,history_actuator_width=4,mode_context_width=2,
        history_steps=26,forecast_steps=50,dt_s=.02,train_windows=int((f.partition=='train').sum()),validation_windows=int((f.partition=='validation').sum()),
        train_flights=f[f.partition=='train'].log_id.nunique(),validation_flights=f[f.partition=='validation'].log_id.nunique(),
        phase_contract=base_manifest['phase_contract'],frequency_contract=base_manifest['frequency_contract'],
        source_parameter_equivalence='recorded selected controls + trim/manual scaling match; complete hardware/firmware path equivalence not asserted',
        model_training_started=False,control_validated=False,sealed_test_accessed=False))
    write(OUT/'completion.json',dict(status='prepared',checked_windows=len(f),rejected_windows=len(rejected),
        outputs={p.name:sha(p) for p in OUT.iterdir() if p.is_file()},training_started=False))
    print(f.groupby(['partition','mode','kind']).size().to_string())
    print('rejected',len(rejected))


if __name__=='__main__':run()
