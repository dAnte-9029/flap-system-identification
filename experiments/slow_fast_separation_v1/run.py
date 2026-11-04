"""Fixed CPU-only feasibility comparison; native target timestamps, train/validation only."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,json,time,subprocess,hashlib,argparse
from pathlib import Path
import numpy as np
import pandas as pd
from methods import cell,fit,causal_template,GRID
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from system_identification.data.pahbd_phase import prepare_phase
DATA=Path('/home/zn/flap-system-identification/dataset/velocity_rate_multistep_v1')
CHANNELS=['vN','vE','vD','p','q','r','vBx']
COLS=['target_velocity_ned_m_s_'+a for a in 'xyz']+['target_angular_velocity_body_rad_s_'+a for a in 'xyz']+['target_velocity_body_m_s_x']
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'docs/analysis/results/slow_fast_separation_v1/fixed_run');args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    contract=json.loads((ROOT/'configs/pahbd/phase_reference.json').read_text());meta={x['flight_id']:x for x in json.loads((DATA/'flight_provenance.json').read_text()) if x['split'] in ('train','validation')}
    protocol=dict(base_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),dataset='velocity_rate_multistep_v1',manifest_sha256=hashlib.sha256((DATA/'dataset_manifest.json').read_bytes()).hexdigest(),splits=['train','validation'],window=51,Q=8,grid='2*pi*(j+0.5)/8',P=3,harmonics=[1,2,3],holdout='run-local cycle index modulo 5 equals 2',minimum_cell_training=8,minimum_each_side=2,mean='midpoint-cell time weights clipped to cycle; identical weights for tau',phase_contract=contract,zero_mean='same native quadrature weights; causal centering uses last completed cycle only',status='running')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));coverage=[];prediction=[];motion=[];runtime=[];events=[];thresholds={};start=time.monotonic();done=0
    for split in ('train','validation'):
        samples=pd.read_parquet(DATA/f'samples_{split}.parquet');phase=pd.read_parquet(DATA/f'native_wing_phase_{split}.parquet')
        controls=['control_flap_motor_normalized','control_left_elevon_normalized','control_right_elevon_normalized','control_rudder_normalized']
        if split=='train':
            for col in controls:
                diffs=[]
                for _,s in samples[samples.sampling_group=='main_50hz'].groupby('flight_id'):
                    d=abs(s[col].diff().to_numpy());diffs.extend(d[np.isfinite(d)&(d>0)])
                thresholds[col]=float(np.quantile(diffs,.9))
            protocol['event_thresholds_train_p90']=thresholds;protocol['event_spacing_s']=2
        for fid,s in samples.groupby('flight_id',sort=False):
            m=meta[fid];done+=1;base=dict(flight_id=fid,split=split,date=m['flight_date']);pt=prepare_phase(phase[phase.flight_id==fid],m['transmission_ratio'],contract)
            ts=pt.timestamp_us.to_numpy()*1e-6;ph=pt.phase_rad.to_numpy();good=pt.phase_usable.to_numpy();cross=np.flatnonzero((np.diff(ph)<-np.pi)&good[:-1]&good[1:])+1
            bounds=ts[cross-1]+(2*np.pi-ph[cross-1])/(ph[cross]+2*np.pi-ph[cross-1])*(ts[cross]-ts[cross-1])
            valid=(s.valid_target_state&~s.excluded&~s.review&s.firmware_verified&~s.provenance_review&s.valid_core).to_numpy()
            for axis,col in enumerate(COLS):
                channel=CHANNELS[axis];tcol='state_measurement_us' if axis<3 or axis==6 else 'target_rate_measurement_us';pubcol='state_publication_us' if axis<3 or axis==6 else 'target_rate_publication_us'
                a=s[[tcol,pubcol,col,'segment_id']].copy();a['valid']=valid;a=a.drop_duplicates(tcol).sort_values(tcol)
                tt=a[tcol].to_numpy()*1e-6;yy=a[col].to_numpy();av=a[pubcol].to_numpy()*1e-6;vv=a.valid.to_numpy()&np.isfinite(yy);seg=a.segment_id.to_numpy()
                runs=[];run=[];missing=0;cells=0;reasons=dict(phase=0,state=0,sparse=0);dur=[];obs=[]
                for k in range(len(bounds)-1):
                    lo,hi=bounds[k:k+2];idx=np.flatnonzero((tt>=lo)&(tt<hi));reason=None
                    if m['sampling_group']!='main_50hz':reason='sparse'
                    elif not good[cross[k]-1:cross[k+1]+1].all():reason='phase'
                    elif any(lo<=r*1e-6<=hi for r in m['reset_times_us']):reason='state'
                    elif len(idx)<2 or not vv[idx].all() or len(np.unique(seg[idx]))!=1 or np.max(np.diff(np.r_[lo,tt[idx],hi]))>.04:reason='state'
                    if reason:
                        reasons[reason]+=1
                        if run:runs.append(run);run=[]
                        continue
                    # Reconstruct only within this QC-admitted cycle, preserving Hall circular anchor.
                    xp=ts[cross[k]-1:cross[k+1]+1];yp=np.unwrap(ph[cross[k]-1:cross[k+1]+1]);pp=np.interp(tt[idx],xp,yp);pp-=2*np.pi*np.floor(np.interp(lo,xp,yp)/(2*np.pi)+1e-7)
                    if np.any(np.diff(pp)<=0) or np.min(pp)<-1e-6 or np.max(pp)>2*np.pi+1e-6:
                        reasons['phase']+=1
                        if run:runs.append(run);run=[]
                        continue

                    if run and run[-1]['segment']!=seg[idx[0]]:runs.append(run);run=[]
                    c=cell(tt[idx],pp,yy[idx],lo,hi);c['segment']=seg[idx[0]];c['availability']=av[idx];c['ready']=max(hi+.02,float(av[idx].max()));c['phase_idx']=(cross[k],cross[k+1]);run.append(c);missing+=np.isnan(c['g']).sum();cells+=8;dur.append(hi-lo);obs.append(len(idx))
                if run:runs.append(run)
                coverage.append(dict(**base,channel=channel,sampling_group=m['sampling_group'],valid_cycles=sum(map(len,runs)),candidate_cycles=max(0,len(bounds)-1),windows_51=sum(max(0,len(r)-50) for r in runs),max_consecutive=max(map(len,runs),default=0),valid_seconds=sum(dur),phase_cell_missing_fraction=missing/cells if cells else np.nan,frequency_p05=np.quantile(1/np.array(dur),.05) if dur else np.nan,frequency_p95=np.quantile(1/np.array(dur),.95) if dur else np.nan,observations_median=np.median(obs) if obs else np.nan,excluded_phase=reasons['phase'],excluded_state=reasons['state'],excluded_sparse=reasons['sparse']))
                for ri,cs in enumerate(runs):
                    if len(cs)<51:continue
                    series=[];synthetic_done=False;initialized=False
                    for n in range(25,len(cs)-25):
                        c=cs[n];baseline,ap,bp,af,bf,d=fit(cs,n)
                        if n%5==2:
                            tr,pa,pb,fa,fb,_=fit(cs,n,True);common=np.isfinite(pb)
                            prediction.append(dict(**base,run=ri,cycle=n,channel=channel,n_observations=len(c['t']),common_observations=int(common.sum()),trend_mse=float(np.mean((tr-c['y'])**2)),A_mse=float(np.mean((pa-c['y'])**2)),B_trend_mse=float(np.mean((pb-fb-c['y'])**2)) if common.all() else np.nan,B_mse=float(np.mean((pb-c['y'])**2)) if common.all() else np.nan,A_fast_rms=float(np.sqrt(np.mean(fa**2))),B_fast_rms=float(np.sqrt(np.mean(fb**2))) if common.all() else np.nan,raw_std=float(np.std(c['y']))))
                        if not np.isfinite(bf).all():continue
                        series.extend(zip(c['t'],c['y'],c['y']-bf,bf))
                        if axis==0 and not synthetic_done:
                            for kind in ('linear','smooth_step'):
                                changed=[];center=c['tau']
                                for z in cs:
                                    delta=.5*(z['t']-center) if kind=='linear' else .3*(lambda u:u*u*(3-2*u))(np.clip((z['t']-center)/.5,0,1))
                                    zz=cell(z['t'],z['p'],z['y']+delta,z['lo'],z['hi']);changed.append(zz)
                                rr=fit(changed,n);motion.append(dict(**base,channel=channel,kind=kind,run=ri,A_fast_leak_rms=float(np.sqrt(np.mean((rr[3]-af)**2))),B_fast_leak_rms=float(np.sqrt(np.mean((rr[4]-bf)**2))),B_weighted_fast_mean=float(c['w']@bf),reconstruction_max=float(np.max(abs(c['y']-((c['y']-bf)+bf))))))
                            synthetic_done=True
                        if n>=51:
                            # History is usable only after all its observations and ending Hall guard arrived.
                            available=[i for i in range(n) if cs[i]['ready']<=c['availability'].min()]
                            if len(available)<51:continue
                            hist=[cs[i] for i in available[-51:]]
                            if available[-1]-available[-51]!=50:continue
                            tick=time.perf_counter_ns();template,slope,anchor=causal_template(hist,n-available[-1]);update=(time.perf_counter_ns()-tick)/1e6
                            if not np.isfinite(template).all():continue
                            estimates=[];times=[]
                            for ti,decision in zip(c['t'],c['availability']):
                                tick=time.perf_counter_ns();j=np.searchsorted(ts+.02,decision,side='right')-1
                                if j<1 or not good[j] or not good[j-1] or decision-ts[j]>.05:estimates.append(np.nan);continue
                                # Delayed Hall extrapolation uses two already available records only.
                                rate=np.angle(np.exp(1j*(ph[j]-ph[j-1])))/(ts[j]-ts[j-1]);query=ph[j]+rate*(ti-ts[j]);estimates.append(np.interp(query,GRID,template,period=2*np.pi));times.append((time.perf_counter_ns()-tick)/1e6)
                            estimates=np.array(estimates);ok=np.isfinite(estimates)
                            runtime.append(dict(**base,channel=channel,run=ri,cycle=n,frequency_hz=1/(c['hi']-c['lo']),history_seconds=hist[-1]['hi']-hist[0]['lo'],history_cycles=51,extrapolation_lead_cycles=n-available[-1],phase_reference_error_p95=float(np.quantile(abs(pt.reference_error_rad.to_numpy()[c['phase_idx'][0]:c['phase_idx'][1]+1]),.95)),fast_slow_difference_rmse=float(np.sqrt(np.mean((estimates[ok]-bf[ok])**2))) if ok.any() else np.nan,teacher_fast_rms=float(np.sqrt(np.mean(bf**2))),initialization_ms=update if not initialized else np.nan,template_update_ms=update,observation_median_ms=np.median(times) if times else np.nan,observation_p95_ms=np.quantile(times,.95) if times else np.nan));initialized=True
                    if series:
                        z=np.array(series);last=-np.inf
                        if not (out/f'curve_{split}_{channel}.csv').exists():
                            pd.DataFrame(z[z[:,0]<z[0,0]+4],columns=['time_s','raw','slow','fast']).to_csv(out/f'curve_{split}_{channel}.csv',index=False)
                        for control in controls:
                            ct=s.timestamp_us.to_numpy()*1e-6;delta=s[control].diff().to_numpy()
                            for ei in np.flatnonzero(abs(delta)>=thresholds[control]):
                                et=ct[ei]
                                if et-last<2 or et<z[0,0]+1 or et>z[-1,0]-1:continue
                                before=(z[:,0]>=et-.5)&(z[:,0]<et);after=(z[:,0]>=et+.5)&(z[:,0]<=et+1)
                                if before.sum()<5 or after.sum()<5:continue
                                diff=z[after,1:].mean(0)-z[before,1:].mean(0);last=et
                                delays=[]
                                for cc in (1,2,3):
                                    target=abs(diff[cc-1])*.5;ii=np.flatnonzero((z[:,0]>=et)&(z[:,0]<=et+1)&(abs(z[:,cc]-z[before,cc].mean())>=target))
                                    delays.append(float(z[ii[0],0]-et) if len(ii) and target>1e-9 else np.nan)
                                events.append(dict(**base,channel=channel,kind='control_event',control=control,event_time_s=et,command_change=delta[ei],raw_change=diff[0],slow_change=diff[1],fast_change=diff[2],raw_half_change_delay_s=delays[0],slow_half_change_delay_s=delays[1],fast_half_change_delay_s=delays[2]))
            elapsed=time.monotonic()-start;print(f'[1-3] flight {done}/{len(meta)} elapsed={elapsed:.1f}s ETA={elapsed/done*(len(meta)-done):.1f}s',flush=True)
            for name,rows in [('coverage',coverage),('cycle_prediction',prediction),('motion_preservation',motion+events),('causal_runtime',runtime)]:pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    protocol['status']='computed';protocol['elapsed_s']=time.monotonic()-start;(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
if __name__=='__main__':main()
