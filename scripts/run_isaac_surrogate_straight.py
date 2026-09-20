"""Straight-line learned-plant experiment in IsaacLab; explicit controller adapter."""
from pathlib import Path
import argparse,json,sys,hashlib,math
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare(output,level=False):
    import pandas as pd
    import numpy as np
    import torch,yaml
    from evaluate_expanded_control_response import load
    from run_main_v2_free_running import make_initial
    from system_identification.training.trajectory_main_v1 import assemble_history_trajectory_windows
    output.mkdir(parents=True,exist_ok=False)
    registry=yaml.safe_load((ROOT/'configs/data/trajectory_dataset_registry.yaml').read_text())
    entry=registry['datasets'][registry['default_dataset_id']];mp=ROOT/entry['manifest_path'];assert sha(mp)==entry['manifest_sha256']
    manifest=json.loads(mp.read_text());root=mp.parent
    for n in ['samples_validation.parquet','windows_validation.parquet']:assert sha(root/n)==manifest['artifact_sha256'][n]
    samples=pd.read_parquet(root/'samples_validation.parquet');windows=pd.read_parquet(root/'windows_validation.parquet')
    if level:
        from system_identification.evaluation.trajectory import QUATERNION_COLUMNS,BODY_RATE_COLUMNS,VELOCITY_COLUMNS
        keep=[]
        lookup=samples.loc[samples.valid_core].set_index(['log_id','segment_id','sample_in_segment']).sort_index()
        for i,row in windows.iterrows():
            state=lookup.loc[(row.log_id,row.segment_id,row.start_sample_in_segment)]
            r,p,_=euler(torch.tensor(state[list(QUATERNION_COLUMNS)].to_numpy(dtype=float)[None]))
            rates=state[list(BODY_RATE_COLUMNS)].to_numpy(dtype=float)
            vel=state[list(VELOCITY_COLUMNS)].to_numpy(dtype=float)
            keep.append(abs(float(r))<math.radians(10) and abs(float(p))<math.radians(30) and abs(rates[2])<.15 and abs(vel[2])<.5 and 5<np.linalg.norm(vel[:2])<12)
        windows=windows.loc[keep]
        if windows.empty:raise ValueError('No level starts')
    windows=pd.concat([g.iloc[[len(g)//2]] for _,g in windows.groupby('log_id')]).reset_index(drop=True)
    batch=assemble_history_trajectory_windows(samples,windows,history_steps=26)
    cp=ROOT/'artifacts/september_expanded_main_v2/model.pt'
    trained=json.loads((ROOT/'docs/analysis/results/september_expanded_main_v2/manifest.json').read_text());assert sha(cp)==trained['checkpoint_sha256']
    sim=load(cp);cases=[]
    for j,row in enumerate(windows.itertuples()):
        state=make_initial(sim,batch,np.array([j]),'warm26','cpu')
        cases.append(dict(log_id=row.log_id,window_id=row.window_id,state=state.snapshot(),command=torch.tensor(batch.trajectory.controls[j:j+1,0],dtype=torch.float32)))
    torch.save(cases,output/'cases.pt')
    report=dict(dataset_id=manifest['dataset_id'],dataset_root=str(root.relative_to(ROOT)),dataset_manifest_sha256=sha(mp),sample_artifact_sha256={n:manifest['artifact_sha256'][n] for n in ['samples_validation.parquet','windows_validation.parquet']},partitions=['validation'],phase_contract=manifest['phase_contract'],frequency_contract=manifest['frequency_contract'],checkpoint=str(cp),checkpoint_sha256=sha(cp),cases_sha256=sha(output/'cases.pt'),selection='middle near-level eligible window per log' if level else 'middle eligible window per log',level_filter='abs roll<10deg, abs pitch<30deg, abs yaw rate<.15rad/s, abs vz<.5m/s, horizontal speed 5..12m/s' if level else None,cases=len(cases),test_opened=False)
    (output/'manifest.json').write_text(json.dumps(report,indent=2));print(report,flush=True)


def euler(q):
    import torch
    w,x,y,z=q.unbind(-1)
    return torch.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),torch.asin(torch.clamp(2*(w*y-z*x),-1,1)),torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))


def native_command(actions,throttle):
    """FLU desired moments -> native FRD servo directions; throttle is TECS effort."""
    import torch
    common=-actions[:,2];diff=-actions[:,3]
    raw=torch.stack((throttle,common+diff,common-diff,-actions[:,1]),1)
    return torch.cat((raw[:,:1].clamp(0,1),raw[:,1:].clamp(-1,1)),1),raw


def run(cases_root,output,headless,flight_chain=False):
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=headless,device='cpu').app
    try:
        import numpy as np
        import pandas as pd
        import torch
        from dataclasses import asdict
        from pxr import UsdGeom,Gf
        from isaaclab.sim import SimulationContext,SimulationCfg
        sys.path.insert(0,'/home/zn/IsaacLab/source/flapping_bot')
        from flapping_bot.px4_like.straight_line_controller import PX4LikeStraightLineController,PX4LikeStraightLineControllerCfg
        from system_identification.integration.isaac_state_backend import IsaacLearnedStateBackend
        from system_identification.models.main_v2_simulator import SimulatorState
        from evaluate_expanded_control_response import load
        torch.set_num_threads(2)
        source=json.loads((cases_root/'manifest.json').read_text());assert sha(cases_root/'cases.pt')==source['cases_sha256'];assert sha(source['checkpoint'])==source['checkpoint_sha256']
        cases=torch.load(cases_root/'cases.pt',map_location='cpu',weights_only=False);sim=load(Path(source['checkpoint']))
        output.mkdir(parents=True,exist_ok=False)
        scene=SimulationContext(SimulationCfg(dt=.02,device='cpu'))
        xform=UsdGeom.Xform.Define(scene.stage,'/World/LearnedAircraft');translate=xform.AddTranslateOp();orient=xform.AddOrientOp()
        body=UsdGeom.Cube.Define(scene.stage,'/World/LearnedAircraft/Body');body.GetSizeAttr().Set(1.);UsdGeom.Xformable(body).AddScaleOp().Set(Gf.Vec3f(.4,.08,.06))
        scene.reset();traces=[];summary=[];configs=[];readback=0.
        source_files=[Path('/home/zn/IsaacLab/source/flapping_bot/flapping_bot/px4_like')/n for n in ['straight_line_controller.py','tecs.py','guidance.py','line_navigation.py']]
        contract=dict(source=source,isaac_runtime=True,model_clock_s=.02,controller_clock_s=.02,duration_s=20.,task='hold initial height, horizontal speed and initial velocity course',wind='zero; horizontal groundspeed treated as airspeed',plant='surrogate owns all dynamics, USD display only, no PhysX vehicle',adapter='TECS throttle directly as motor; native common=-FLU pitch, differential=-FLU roll, rudder=-FLU yaw; no frequency-to-motor conversion',controller='existing PX4-like defaults, no gain tuning; trim/initial actions from initial state',termination=dict(height_error_m=5,cross_track_m=10,speed_error_m_s=5,roll_deg=80,pitch_deg=60),termination_note='diagnostic stop limits fixed before execution, not certified safety limits',test_opened=False,source_sha256={str(p):sha(p) for p in source_files},script_sha256=sha(__file__))
        if flight_chain:
            from system_identification.integration.px4_flight_straight import FlightStraightAdapter
            native_root=ROOT/'artifacts/px4_e624_outer'
            parameters_path=ROOT/'docs/analysis/results/flight_chain_alignment/flight_parameters.json'
            contract.update(controller='e624 baseline A, FLAP_SLOW_EN=0, native TECS/NPFG, flight attitude/rate/allocation',
                controller_clock_s=.0025,outer_attitude_clock_s=.02,rate_sensor_contract='8 rate updates per model step, held ideal observations; last command applied at model 50 Hz',
                adapter='FRD native torque -> audited unnormalized allocation; no second PWM reversal',
                initialization='native TECS reset; rate I=0, gain=1, speed filter=current horizontal speed; no logged hidden controller state',
                limitations=['zero wind and sea-level density','ideal nav quality factor=1','no takeoff/landing/manual handover','no B2b','D gains zero, no gyro derivative emulation'],
                native_manifest_sha256=sha(native_root/'manifest.json'),parameters_sha256=sha(parameters_path))
            for name in ['control/px4_flight_chain.py','control/px4_native_outer.py','control/px4_pitch.py','integration/px4_flight_straight.py']:
                path=ROOT/'src/system_identification'/name;contract['source_sha256'][str(path)]=sha(path)
        (output/'manifest.json').write_text(json.dumps(contract,indent=2))
        for cid,case in enumerate(cases):
            for mode in ['closed','open_hold']:
                backend=IsaacLearnedStateBackend(sim,checkpoint_id=source['checkpoint_sha256'])
                initial=SimulatorState.from_snapshot(case['state'],device='cpu');obs=backend.reset(state=initial)
                p0=obs.position_w.clone();v0=obs.velocity_w.clone();heading=float(torch.atan2(v0[0,1],v0[0,0]));speed=float(torch.linalg.norm(v0[0,:2]));height=float(p0[0,2]);r,p,y=euler(obs.quaternion_wb)
                u0=case['command'];common=float((u0[0,1]+u0[0,2])/2);diff=float((u0[0,1]-u0[0,2])/2)
                cfg=PX4LikeStraightLineControllerCfg(control_dt_s=.02,line_start_xy=tuple(p0[0,:2].tolist()),line_end_xy=(float(p0[0,0])+1000*math.cos(heading),float(p0[0,1])+1000*math.sin(heading)),height_sp_m=height,speed_sp_mps=speed,enable_speed_hold=True,pitch_trim_deg=-math.degrees(float(p)),freq_trim_hz=3+float(u0[0,0])*1.6,initial_elevon_pitch_action=-common,initial_elevon_roll_action=-diff)
                if flight_chain:
                    controller=FlightStraightAdapter(native_root,parameters_path,position_n=initial.position_n[0].numpy(),velocity_n=initial.velocity_n[0].numpy())
                else:
                    controller=PX4LikeStraightLineController(cfg,device=torch.device('cpu'));controller.reset()
                if mode=='closed':configs.append(dict(case=cid,log_id=case['log_id'],config=controller.outer.p if flight_chain else asdict(cfg)))
                reason='completed';start=len(traces)
                for k in range(1000):
                    r,p,y=euler(obs.quaternion_wb)
                    actions,diag=controller.compute_actions(pos_local=obs.position_w,ground_vel_local=obs.velocity_w,roll=r,pitch=p,yaw=y,ang_vel_body=obs.angular_velocity_b)
                    u,raw=native_command(actions,diag['tecs_throttle_sp'])
                    if flight_chain:raw=diag['allocation_raw']
                    if mode=='open_hold':u=u0.clone()
                    obs,_=backend.step(u)
                    if not all(torch.isfinite(v).all() for v in [obs.position_w,obs.velocity_w,obs.quaternion_wb,obs.angular_velocity_b]):reason='nonfinite';break
                    pos=obs.position_w[0].tolist();q=obs.quaternion_wb[0].tolist();translate.Set(Gf.Vec3d(*pos));orient.Set(Gf.Quatf(q[0],Gf.Vec3f(*q[1:])))
                    back=translate.Get();readback=max(readback,max(abs(back[j]-pos[j]) for j in range(3)))
                    if not headless:scene.render()
                    elif k%50==0:app.update()
                    r,p,y=euler(obs.quaternion_wb);delta=obs.position_w-p0
                    cross=-math.sin(heading)*float(delta[0,0])+math.cos(heading)*float(delta[0,1]);he=float(obs.position_w[0,2])-height;sp=float(torch.linalg.norm(obs.velocity_w[0,:2]));se=sp-speed
                    traces.append(dict(
                        case=cid,log_id=case['log_id'],mode=mode,time_s=(k+1)*.02,
                        height_error_m=he,cross_track_m=cross,speed_error_m_s=se,speed_m_s=sp,
                        roll_deg=math.degrees(float(r)),pitch_deg=math.degrees(float(p)),
                        x=pos[0],y=pos[1],z=pos[2],motor=float(u[0,0]),left=float(u[0,1]),
                        right=float(u[0,2]),rudder=float(u[0,3]),
                        surface_limit=bool((u[:,1:].abs()>=.999).any()),
                        motor_limit=bool(((u[:,:1]<=.001)|(u[:,:1]>=.999)).any()),
                        mix_clipped=bool((raw[:,1:].abs()>1).any()) if mode=='closed' else False,
                        flight_chain_diagnostics=json.dumps(diag['flight_chain']) if flight_chain else None))
                    failures=[n for n,v,limit in [('height_error',abs(he),5),('cross_track',abs(cross),10),('speed_error',abs(se),5),('roll',abs(math.degrees(float(r))),80),('pitch',abs(math.degrees(float(p))),60)] if v>limit]
                    if failures:reason='+'.join(failures);break
                if flight_chain:controller.close()
                f=pd.DataFrame(traces[start:]);summary.append(dict(case=cid,log_id=case['log_id'],mode=mode,termination=reason,duration_s=(k+1)*.02,height_rmse=float(np.sqrt(np.mean(f.height_error_m**2))) if len(f) else None,cross_track_rmse=float(np.sqrt(np.mean(f.cross_track_m**2))) if len(f) else None,speed_rmse=float(np.sqrt(np.mean(f.speed_error_m_s**2))) if len(f) else None,surface_limit_fraction=float(f.surface_limit.mean()) if len(f) else None,motor_limit_fraction=float(f.motor_limit.mean()) if len(f) else None))
                pd.DataFrame(summary).to_csv(output/'summary.csv',index=False);print(summary[-1],flush=True)
        pd.DataFrame(traces).to_csv(output/'traces.csv',index=False);(output/'controller_configs.json').write_text(json.dumps(configs,indent=2));(output/'verification.json').write_text(json.dumps(dict(completed=True,isaac_runtime=True,cases=len(cases),runs=len(summary),scene_position_readback_error=readback,completed_20s=sum(s['termination']=='completed' for s in summary),test_opened=False),indent=2));print('EXPERIMENT_COMPLETE',flush=True)
    finally:
        app.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--level',action='store_true');p.add_argument('--cases',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--headless',action='store_true');p.add_argument('--flight-chain',action='store_true');a=p.parse_args()
    if a.prepare:prepare(a.output,a.level)
    else:run(a.cases,a.output,a.headless,a.flight_chain)
if __name__=='__main__':main()
