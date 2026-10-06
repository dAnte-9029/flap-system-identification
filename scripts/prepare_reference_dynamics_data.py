"""Prepare flight-isolated reference-driven windows; no model fitting/training."""
from pathlib import Path
import hashlib
import json
import subprocess

import numpy as np
import pandas as pd
import yaml
from pyulog import ULog
from scipy.spatial.transform import Rotation

from audit_control_reference_semantics import past_values
from mine_existing_reference_excitation import euler, runs

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/analysis/results/reference_dynamics_data_v1'
CACHE=ROOT/'dataset/reference_dynamics_data_v1'
REG=ROOT/'configs/data/trajectory_dataset_registry.yaml'
HISTORY=26
FUTURE=50
STRIDE=10
TOPICS=['vehicle_status','vehicle_control_mode','vehicle_land_detected','vehicle_attitude_setpoint',
        'vehicle_attitude','vehicle_angular_velocity','vehicle_local_position','actuator_servos',
        'actuator_motors','wing_phase','airspeed_validated','position_setpoint_triplet']
PREFIXES=('FW_','CA_','PWM_','DSHOT_','FLAP_','IMU_','SENS_BOARD_')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def configuration_id(firmware,parameters):
    return hashlib.sha256(json.dumps([firmware,parameters],sort_keys=True).encode()).hexdigest()


def chunks(valid, mode, reset):
    """Each block belongs to one mode, cannot straddle a reset or stale sample."""
    out=[]
    for lo,hi in runs(valid):
        edges=np.flatnonzero((mode[lo+1:hi]!=mode[lo:hi-1])|reset[lo+1:hi])+lo+1
        bounds=[lo,*edges,hi]
        for a,b in zip(bounds[:-1],bounds[1:]):
            # Fixed 0.5s trim around all boundaries, independent of observed outcome.
            if b-a>=HISTORY+FUTURE+50:out.append((a+25,b-25))
    return out


def window_origins(start,stop):
    return range(start+HISTORY,stop-FUTURE,STRIDE)


def run():
    OUT.mkdir(parents=True,exist_ok=False);CACHE.mkdir(parents=True,exist_ok=False)
    reg=yaml.safe_load(REG.read_text());did=reg['default_dataset_id'];entry=reg['datasets'][did]
    mp=ROOT/entry['manifest_path'];assert sha(mp)==entry['manifest_sha256']
    parent=json.loads(mp.read_text());assign=parent['split_contract']['assignments']
    allowed={p:s for s in ['train','validation'] for p in assign[s]}
    denied={Path(p).name for s,ps in assign.items() if s not in ['train','validation'] for p in ps}
    assert not any(Path(p).name in denied for p in allowed)
    protocol=dict(dataset_id='reference_dynamics_data_v1',parent_id=did,parent_manifest_sha256=sha(mp),
        registry_sha256=sha(REG),branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        script_sha256=sha(__file__),helper_sha256={p:sha(ROOT/p) for p in ['scripts/audit_control_reference_semantics.py','scripts/mine_existing_reference_excitation.py']},
        allowed=allowed,partitions=['train','validation'],sealed_test_accessed=False,
        model_boundary='attitude/rate inner controllers + allocation + actuator + aircraft; no navigation outer loop',
        future_input=['roll_reference_rad','pitch_reference_rad','yaw_sp_move_rate_rad_s','thrust_body_x'],
        forbidden_future=['measured attitude/rates/velocity','actuator commands','airspeed','wing phase/frequency','controller feedback','actual future nav references as counterfactual policy evidence'],
        interpretation='logged future references are conditional predictor inputs, not proof of closed-loop replanning performance',
        history_steps=HISTORY,forecast_steps=FUTURE,stride_steps=STRIDE,dt_s=.02,
        phase_contract=parent['phase_contract'],frequency_contract=parent['frequency_contract'],
        timing='last sample whose publication and sample timestamps<=query; no interpolation; per-topic age arrays retained',
        screening='armed airborne no failsafe, nav3 or15 with matching auto/manual flags and attitude/rates enabled; 60ms dynamic freshness; 1.2s status freshness; valid velocity and phase; quaternion norm0.9..1.1; nonzero thrust_y/z excluded; parameter-changing logs excluded',
        boundaries='invalid samples, mode changes and attitude/velocity reset changes split; trim0.5s both ends',
        excitation='tag known1s reference variation: roll/pitch range>=1deg OR yaw-rate range>=0.05rad/s OR thrust range>=0.03; also retain quiet windows; no output-amplitude or response-sign gate',
        config_groups='exact firmware string plus all present FW_/CA_/PWM_/DSHOT_/FLAP_/IMU_/SENS_BOARD_ initial values; absence differs; full parameter snapshot also retained; not proof of complete hardware equality',
        primary_group='largest eligible train window count among groups with>=2 train flights; validation availability reported after selection, never used for fitting or ranking',
        normalization='unique past/current rows and known reference rows used by primary train windows only; phase normalized relative to each origin in loader, sin/cos left unscaled',
        no_training=True)
    write(OUT/'protocol.json',protocol)
    windows=[];segments=[];logs=[];rejections=[];sourcepins={};cachepins={}
    for log,split in sorted(allowed.items()):
        path=Path('/home/zn/QgcLogs')/log;assert sha(path)==parent['source']['ulog_sha256'][log]
        sourcepins[log]=dict(path=str(path),sha256=sha(path),partition=split)
        u=ULog(str(path),message_name_filter_list=TOPICS)
        ds={d.name:d.data for d in u.data_list if d.multi_id==0}
        firmware=str(u.msg_info_dict.get('ver_sw','unknown'))
        pars={k:float(v) for k,v in u.initial_parameters.items() if k.startswith(PREFIXES)}
        group=configuration_id(firmware,pars);logkey=hashlib.sha256(log.encode()).hexdigest()[:16]
        write(OUT/f'parameters_{logkey}.json',dict(log_id=log,firmware=firmware,group=group,
            grouped=pars,all_initial={k:float(v) for k,v in u.initial_parameters.items()},
            parameter_changes=[str(x) for x in u.changed_parameters]))
        refdata=ds['vehicle_attitude_setpoint'];times=np.arange(int(refdata['timestamp'][0]),int(refdata['timestamp'][-1]),20000,dtype=np.int64)
        ages={}
        def get(topic,keys,limit=.06):
            if topic not in ds or not all(k in ds[topic] for k in keys):
                ages[topic]=np.full(len(times),np.inf)
                return np.full((len(times),len(keys)),np.nan),np.zeros(len(times),bool)
            x,a,b=past_values(ds[topic],keys,times);ages[topic]=np.maximum(a,b)
            return x,np.isfinite(x).all(1)&(np.maximum(a,b)<=limit)
        rq,refok=get('vehicle_attitude_setpoint',[f'q_d[{j}]' for j in range(4)]+['yaw_sp_move_rate','thrust_body[0]','thrust_body[1]','thrust_body[2]'])
        refs=np.column_stack([np.deg2rad(euler(rq[:,:4])),rq[:,4:6]])
        st,stok=get('vehicle_status',['nav_state','failsafe'],1.2)
        mo,mook=get('vehicle_control_mode',['flag_armed','flag_control_manual_enabled','flag_control_auto_enabled','flag_control_attitude_enabled','flag_control_rates_enabled','flag_control_climb_rate_enabled'],1.2)
        land,landok=get('vehicle_land_detected',['landed'],1.2)
        q,qok=get('vehicle_attitude',[f'q[{j}]' for j in range(4)]+['quat_reset_counter'])
        rate,rateok=get('vehicle_angular_velocity',[f'xyz[{j}]' for j in range(3)])
        vel,velok=get('vehicle_local_position',['vx','vy','vz','v_xy_valid','v_z_valid','vxy_reset_counter','vz_reset_counter','x','y','z'])
        servo,servook=get('actuator_servos',[f'control[{j}]' for j in range(3)])
        motor,motorok=get('actuator_motors',['control[0]'])
        phase,phaseok=get('wing_phase',['phase_rad','flap_frequency_hz','phase_valid'])
        air,airok=get('airspeed_validated',['true_airspeed_m_s','airspeed_source'],.3)
        trip,tripok=get('position_setpoint_triplet',['current.valid','current.type'],1.2)
        attitude_norm=np.linalg.norm(q[:,:4],axis=1)
        refnorm=np.linalg.norm(rq[:,:4],axis=1)
        stabil=(st[:,0]==15)&(mo[:,1]==1)&(mo[:,5]==0)
        mission=(st[:,0]==3)&(mo[:,2]==1)
        flags=stok&mook&landok&(mo[:,0]==1)&(mo[:,3]==1)&(mo[:,4]==1)&(land[:,0]==0)&(st[:,1]==0)
        masks=dict(mode_airborne=flags&(stabil|mission),reference=refok&np.isfinite(refs).all(1)&(refnorm>.9)&(refnorm<1.1)&(np.abs(rq[:,6:8])<1e-6).all(1),
            attitude=qok&(attitude_norm>.9)&(attitude_norm<1.1),rates=rateok,
            velocity=velok&(vel[:,3:5]==1).all(1),commands=servook&motorok,
            phase=phaseok&(phase[:,2]==1)&(phase[:,1]>0))
        valid=np.logical_and.reduce(list(masks.values()))
        if len(u.changed_parameters):valid[:]=False
        for k,v in masks.items():rejections.append(dict(log_id=log,partition=split,reason=k,failed_rows=int((~v).sum()),total_rows=len(times)))
        reset=np.r_[False,((np.diff(q[:,4])!=0)|(np.diff(vel[:,5])!=0)|(np.diff(vel[:,6])!=0))]
        safeq=q[:,:4].copy();bad=~np.isfinite(safeq).all(1)|(np.linalg.norm(safeq,axis=1)<.5);safeq[bad]=[1,0,0,0]
        rot=Rotation.from_quat(safeq[:,[1,2,3,0]])
        bodyvel=rot.inv().apply(vel[:,:3]);gravity=rot.inv().apply(np.tile([0.,0.,1.],(len(times),1)))
        airvalid=airok&(air[:,1]>0)&(air[:,0]>0)
        state=np.column_stack([bodyvel,rate,gravity,np.sin(phase[:,0]),np.cos(phase[:,0]),phase[:,1],np.where(airvalid,air[:,0],0),airvalid.astype(float)])
        data=dict(timestamp_us=times,state_features=state.astype(np.float32),references=refs.astype(np.float32),
            actuator_commands=np.column_stack([motor,servo]).astype(np.float32),quaternion=q[:,:4].astype(np.float32),
            velocity_n=vel[:,:3].astype(np.float32),position_n=vel[:,7:10].astype(np.float32),rates_b=rate.astype(np.float32),
            phase_rad=phase[:,0].astype(np.float32),valid=valid,reset=reset,nav_state=st[:,0],
            mission_type=np.where(tripok&(trip[:,0]==1),trip[:,1],-1),
            age_topic_names=np.array(list(ages)),age_s=np.column_stack(list(ages.values())).astype(np.float32))
        dest=CACHE/f'{logkey}.npz';np.savez_compressed(dest,**data);cachepins[str(dest.relative_to(ROOT))]=sha(dest)
        count=0
        for seg,(a,b) in enumerate(chunks(valid,st[:,0],reset)):
            typ=data['mission_type'][a:b];mode=int(st[a,0]);kind='stabilized' if mode==15 else ('mission_position' if (typ==0).all() else 'mission_loiter' if (typ==2).all() else 'mission_mixed_or_unknown')
            sid=f'{logkey}_{seg}'
            segments.append(dict(segment_id=sid,log_id=log,partition=split,group=group,mode=mode,kind=kind,start_index=a,stop_index=b,start_us=int(times[a]),end_us=int(times[b-1]),duration_s=(b-a)*.02))
            for i in window_origins(a,b):
                tape=refs[i:i+FUTURE];variation=np.ptp(tape,axis=0)
                active=bool((variation[:2]>=np.deg2rad(1)).any() or variation[2]>=.05 or variation[3]>=.03)
                windows.append(dict(window_id=f'{sid}_{i}',log_id=log,partition=split,group=group,segment_id=sid,mode=mode,kind=kind,
                    cache=str(dest.relative_to(ROOT)),origin_index=i,origin_timestamp_us=int(times[i]),active_reference=active,
                    roll_range_rad=float(variation[0]),pitch_range_rad=float(variation[1]),yaw_rate_range=float(variation[2]),thrust_range=float(variation[3]),
                    command_near_limit=bool((np.abs(servo[i:i+FUTURE])>=.98).any()),
                    initial_groundspeed=float(np.linalg.norm(vel[i,:2])),initial_vz=float(vel[i,2]),initial_flap_frequency=float(phase[i,1])))
                count+=1
        logs.append(dict(log_id=log,partition=split,group=group,firmware=firmware,windows=count,valid_s=float(valid.sum()*.02),parameter_changes=len(u.changed_parameters)))
        print(len(logs),len(allowed),log,count,flush=True)
    w=pd.DataFrame(windows);s=pd.DataFrame(segments);l=pd.DataFrame(logs)
    w.to_csv(OUT/'windows.csv',index=False);s.to_csv(OUT/'segments.csv',index=False);l.to_csv(OUT/'logs.csv',index=False)
    pd.DataFrame(rejections).to_csv(OUT/'quality_counts.csv',index=False)
    assert not set(w[w.partition=='train'].log_id)&set(w[w.partition=='validation'].log_id)
    ranking=w[w.partition=='train'].groupby('group').agg(windows=('window_id','size'),flights=('log_id','nunique')).query('flights>=2').sort_values(['windows','group'],ascending=[False,True])
    assert len(ranking),'No group with two train flights'
    primary=str(ranking.index[0]);selected=w[w.group==primary].copy();selected.to_csv(OUT/'primary_windows.csv',index=False)
    # Normalization uses union of rows belonging to TRAIN windows, without overlap weighting.
    state_rows=[];reference_rows=[];command_rows=[]
    for cache,items in selected[selected.partition=='train'].groupby('cache'):
        with np.load(ROOT/cache,allow_pickle=False) as z:
            hidx=np.unique(np.concatenate([np.arange(i-HISTORY+1,i+1) for i in items.origin_index]))
            ridx=np.unique(np.concatenate([np.arange(i-HISTORY+1,i+FUTURE) for i in items.origin_index]))
            state_rows.append(z['state_features'][hidx]);reference_rows.append(z['references'][ridx]);command_rows.append(z['actuator_commands'][hidx-1])
    stats={}
    for name,arrays in [('state',state_rows),('reference',reference_rows),('past_actuator',command_rows)]:
        x=np.concatenate(arrays).astype(float);assert np.isfinite(x).all()
        stats[name+'_mean']=x.mean(0);stats[name+'_scale']=x.std(0).clip(1e-6)
    stats['state_mean'][9:11]=0;stats['state_scale'][9:11]=1
    norm=CACHE/'normalization.npz';np.savez(norm,**stats);cachepins[str(norm.relative_to(ROOT))]=sha(norm)
    validation=selected[selected.partition=='validation']
    write(OUT/'selection.json',dict(primary_group=primary,train_selection_only=True,
        train_windows=int((selected.partition=='train').sum()),validation_windows=len(validation),
        train_flights=selected[selected.partition=='train'].log_id.nunique(),validation_flights=validation.log_id.nunique(),
        paired_validation_available=bool(len(validation)),normalization_path=str(norm.relative_to(ROOT)),
        ready_for_training_preflight=bool(len(validation)),control_validated=False))
    manifest=dict(dataset_id='reference_dynamics_data_v1',parent_manifest_sha256=sha(mp),phase_contract=parent['phase_contract'],frequency_contract=parent['frequency_contract'],
        sources=sourcepins,cache_sha256=cachepins,primary_group=primary,
        tables_sha256={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'windows.csv',OUT/'segments.csv',OUT/'primary_windows.csv']},
        protocol_sha256=sha(OUT/'protocol.json'),sealed_test_accessed=False,model_training_started=False)
    write(CACHE/'manifest.json',manifest)
    write(OUT/'completion.json',dict(status='complete',logs=len(logs),windows=len(w),segments=len(s),manifest=str((CACHE/'manifest.json').relative_to(ROOT)),manifest_sha256=sha(CACHE/'manifest.json'),
        outputs={p.name:sha(p) for p in OUT.iterdir() if p.is_file()},sealed_test_accessed=False,model_training_started=False))


if __name__=='__main__':run()
