"""Single-factor slow-rate objective comparison; frozen 3-seed update budget."""
import json
import time
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from train_reference_unit_gravity import ART,model_from
from control_windows_v3 import predict
from audit_reference_v4 import OUT
from system_identification.models.reference_gru_unit_gravity import ReferenceGRUUnitGravity
from system_identification.models.reference_rate_readout import causal_average_rates
from evaluate_control_windows_v2 import error_rows,summarize
DEST=OUT/'slow_objective'


def slow_loss(pred,target,model,history):
    rate=((causal_average_rates(history,pred['rates_b'])-causal_average_rates(history,target['rates_b']))/model.rate_scale).square().mean()
    velocity=((pred['velocity_n']-target['velocity_n'])/model.velocity_scale).square().mean()
    dot=(pred['quaternion']*target['quaternion']).sum(-1).clamp(-1,1)
    return rate+velocity+.2*(4*(1-dot.square()).clamp_min(0)/.01).mean()


def evaluate(model,x,y,w,label,seed):
    frames=[];waves=[]
    for partition in ['train','validation']:
        meta=w[w.partition==partition];ix=meta.array_index.to_numpy();truth={k:v[ix].numpy() for k,v in y.items()}
        pp=predict(model,x,ix);history=x['history_state'][ix,:,3:6]
        np.savez_compressed(DEST/f'{label}_{seed}_{partition}_predictions.npz',array_index=ix,**pp)
        for readout in ['instantaneous','causal_mean_200ms']:
            p={k:v.copy() for k,v in pp.items()};t={k:v.copy() for k,v in truth.items()}
            if readout!='instantaneous':
                p['rates_b']=causal_average_rates(history,torch.from_numpy(p['rates_b'])).numpy()
                t['rates_b']=causal_average_rates(history,torch.from_numpy(t['rates_b'])).numpy()
            ef=error_rows(p,t,meta,label,seed);ff=summarize(ef);ff['partition']=partition;ff['readout']=readout;frames.append(ff)
            if partition=='validation':
                for horizon in [25,50]:
                    initial=(history[:,-10:].mean(1) if readout!='instantaneous' else history[:,-1]).numpy()
                    for ax,axis in enumerate(['p','q','r']):
                        actual=t['rates_b'][:,:horizon,ax]-initial[:,None,ax];sim=p['rates_b'][:,:horizon,ax]-initial[:,None,ax]
                        f=meta[['array_index','log_id','roll_range_rad','pitch_range_rad']].copy();f['arm']=label;f['seed']=seed;f['readout']=readout;f['axis']=axis;f['horizon_s']=horizon*.02
                        f['waveform_rmse']=np.sqrt(np.mean((actual-sim)**2,axis=1));f['amplitude_error']=abs(abs(actual).max(1)-abs(sim).max(1))
                        f['peak_time_error_s']=.02*abs(np.argmax(abs(actual),axis=1)-np.argmax(abs(sim),axis=1))
                        f['observed_increment_peak']=abs(actual).max(1);waves.append(f)
    return pd.concat(frames),pd.concat(waves)


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    assert json.loads((OUT/'audit_status.json').read_text())['status']=='complete'
    DEST.mkdir(exist_ok=False);x,y,stats=base.load();w=pd.read_csv(base.DATA/'windows_v2.csv')
    cohort=pd.read_csv(base.ROOT/'docs/analysis/results/control_windows_v3/unit_gravity/arm_A.csv');assert (cohort.partition=='train').all()
    indices=cohort.array_index.to_numpy();prob=cohort.sampling_probability.to_numpy();prob/=prob.sum()
    protocol=dict(parent_protocol_sha256=base.sha(OUT/'protocol.json'),source_manifest_sha256=base.sha(base.DATA/'manifest.json'),
        script_sha256=base.sha(__file__),readout_sha256=base.sha(base.ROOT/'src/system_identification/models/reference_rate_readout.py'),
        decision='No harmonic candidate triggered. Separate follow-up after training-spectrum/capacity audit: change only rate loss to causal10-sample average.',
        target='velocity_NED and attitude at forecast time; rates averaged over current and previous9 samples',
        filter_history='observed last9 raw rates before forecast, then predicted raw latent rates; truth uses identical causal filter',
        linear_phase_delay_s=.09,not_equivalent_to_instantaneous_state=True,
        seeds=[17,23,42],updates=1600,batch=32,optimizer='AdamW3e-4 wd1e-4 clip1',
        normalization='unchanged A train only',dataset='same full407 train windows and fixed739 development-validation windows',
        comparison='A1600 frozen vs slow-loss1600, BOTH through identical causal readout; raw metrics secondary',
        primary='equal-flight active p and q RMSE at0.5/1s, equal axes/horizons/seeds',
        relative_gate='>=5percent primary gain, everyseed improves, no all-window metric >5percent worse at0.5/1s',
        absolute_gate='eachseed0.5s active averaged p/q<=.15rad/s; attitude<=3deg; velocity<=.5m/s',
        limits='research thresholds not validated flight safety; filtered p/q pass cannot certify instantaneous control; validation already viewed',
        reference_response='held-reference ablation + nominal0.5/1s waveforms; no measured counterfactual certification; long rollouts stress only',
        sealed_test_accessed=False)
    base.write(DEST/'protocol.json',protocol);frames=[];waves=[];completed=[]
    for seed in [17,23,42]:
        torch.manual_seed(seed);rng=np.random.default_rng(seed);m=ReferenceGRUUnitGravity(stats)
        opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);curve=[];started=time.monotonic()
        base.write(DEST/'status.json',dict(status='running',seed=seed,completed=completed))
        for step in range(1,1601):
            ix=indices[rng.choice(len(indices),32,replace=True,p=prob)]
            opt.zero_grad(set_to_none=True);loss=slow_loss(m(**{k:v[ix] for k,v in x.items()}),{k:v[ix] for k,v in y.items()},m,x['history_state'][ix,:,3:6])
            assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
            if step==1 or step%100==0:
                row=dict(seed=seed,step=step,loss=float(loss.detach()),gradient_norm=float(norm),elapsed_s=time.monotonic()-started);curve.append(row);print(json.dumps(row),flush=True)
        path=DEST/f'slow_{seed}.pt';torch.save(dict(state_dict=m.state_dict(),feature_contract='unit_gravity_v1',objective='causal_average_rates10',protocol_sha256=base.sha(DEST/'protocol.json'),seed=seed,steps=1600),path)
        pd.DataFrame(curve).to_csv(DEST/f'training_{seed}.csv',index=False)
        for name,model in [('slow',m.eval()),('A1600',model_from(ART/f'A_long_{seed}/final.pt'))]:
            f,wa=evaluate(model,x,y,w,name,seed);frames.append(f);waves.append(wa)
        # Reference ablation scored only on the SAME slow target; no causal-effect interpretation.
        ev=w[w.partition=='validation'];ix=ev.array_index.to_numpy();truth={k:v[ix].numpy() for k,v in y.items()};hist=x['history_state'][ix,:,3:6]
        hp=predict(m,x,ix,hold=True)
        for d in [hp,truth]:d['rates_b']=causal_average_rates(hist,torch.from_numpy(d['rates_b'])).numpy()
        f=summarize(error_rows(hp,truth,ev,'slow_reference_held',seed));f['partition']='validation';f['readout']='causal_mean_200ms';frames.append(f)
        completed.append(dict(seed=seed,checkpoint_sha256=base.sha(path)))
        pd.concat(frames).to_csv(DEST/'per_flight.csv',index=False);pd.concat(waves).to_csv(DEST/'nominal_waveforms.csv',index=False)
    base.write(DEST/'status.json',dict(status='complete',completed=completed,sealed_test_accessed=False))

if __name__=='__main__':main()
