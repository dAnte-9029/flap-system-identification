#!/usr/bin/env python3
"""Matched small P0/P1/P2/oracle probes; oracle never becomes a deployable input."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json,argparse,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd,torch
from train_main_v2_step4_probes import phase_history
from prepare_main_v2_phase_reference import OLD,DATA
from run_main_v2_free_running import sha


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);pa.add_argument('--artifacts',type=Path,required=True);pa.add_argument('--device',default='cuda:1');a=pa.parse_args();out=a.output
    if a.artifacts.exists() and any(a.artifacts.iterdir()):raise FileExistsError(a.artifacts)
    a.artifacts.mkdir(parents=True)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    protocol=json.loads((out/'protocol.json').read_text());selection=json.loads((out/'selected_estimator.json').read_text());templates=json.loads((out/'canonical_template.json').read_text())
    meta={s:pd.read_csv(OLD/f'{s}_points.csv') for s in ['train','validation']};arrays={s:np.load(OLD/f'{s}_diagnostic_arrays.npz') for s in meta}
    estimates=pd.read_csv(out/'phase_offset_estimation.csv');oracle=pd.read_csv(out/'oracle_offsets_NOT_DEPLOYABLE.csv')
    inputs={};durations={}
    for split,m in meta.items():
        samples=pd.read_parquet(DATA/f'samples_{split}.parquet');times=samples[samples.valid_core].set_index(['log_id','segment_id','sample_in_segment']).timestamp_us
        future=times.loc[[(x.log_id,x.segment_id,x.sample_in_segment+2) for x in m.itertuples()]].to_numpy()*1e-6
        durations[split]=future-m.timestamp_s.to_numpy()
        raw=arrays[split]['feature_history26'].astype(np.float32)
        e=estimates[(estimates.split==split)&(estimates.method==selection['method'])&(estimates.requested_history_steps==selection['history_steps'])].set_index('point_index').loc[np.arange(len(m))]
        offsets=m.log_id.map(oracle[oracle.split==split].set_index('log_id').offset_rad).to_numpy()
        inputs[split]=dict(P0=raw,P1=phase_history(raw,m.phase.to_numpy()),P2=phase_history(raw,m.phase.to_numpy()+e.offset_rad.to_numpy()),P_oracle=phase_history(raw,m.phase.to_numpy()+offsets))
    fit=np.flatnonzero(meta['train'].timestamp_s.to_numpy()<meta['train'].log_id.map(templates['cuts']).to_numpy()-2)[::2]
    hold=np.flatnonzero(meta['train'].timestamp_s.to_numpy()>meta['train'].log_id.map(templates['cuts']).to_numpy()+2)
    mean=inputs['train']['P0'][fit].mean(0);std=inputs['train']['P0'][fit].std(0);std[std<1e-5]=1
    cols=[12*i+j for i in range(26) for j in [9,10]];mean[cols]=0;std[cols]=1
    ym=arrays['train']['target_D0_raw'][fit].mean(0);ys=arrays['train']['target_D0_raw'][fit].std(0)
    metrics=[];training=[]
    for target in protocol['probe_targets']:
        for seed in protocol['seeds']:
            for contract in protocol['probe_contracts']:
                started=time.monotonic();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                model=torch.nn.Sequential(torch.nn.Linear(420,64),torch.nn.SiLU(),torch.nn.Linear(64,64),torch.nn.SiLU(),torch.nn.Linear(64,6)).to(a.device)
                opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-5);gen=torch.Generator().manual_seed(seed)
                xt=torch.tensor((inputs['train'][contract][fit]-mean)/std,dtype=torch.float32,device=a.device);yt=torch.tensor((arrays['train']['target_'+target][fit]-ym)/ys,dtype=torch.float32,device=a.device)
                for epoch in range(30):
                    ix=torch.randperm(len(fit),generator=gen)
                    for start in range(0,len(fit),512):
                        j=ix[start:start+512].to(a.device);opt.zero_grad(set_to_none=True);loss=(model(xt[j])-yt[j]).square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
                model.eval();preds={}
                with torch.inference_mode():
                    for scope,split,idx in [('train_temporal_holdout','train',hold),('validation','validation',np.arange(len(meta['validation'])))]:
                        pred=np.concatenate([model(torch.tensor((inputs[split][contract][j]-mean)/std,dtype=torch.float32,device=a.device)).cpu().numpy() for j in np.array_split(idx,max(1,int(np.ceil(len(idx)/4096))))])*ys+ym;preds[scope]=pred
                        for evaltarget in ['D0_raw','increment2','increment2_state']:
                            truth=arrays[split]['target_'+('increment2' if evaltarget=='increment2_state' else evaltarget)][idx]
                            for log in meta[split].iloc[idx].log_id.unique():
                                select=meta[split].iloc[idx].log_id.to_numpy()==log
                                for signal,sl in [('linear',slice(0,3)),('angular',slice(3,6)),('q_dot',slice(4,5)),('az_n',slice(2,3))]:
                                    error=pred[select,sl]-truth[select,sl]
                                    if evaltarget=='increment2_state':error=error*durations[split][idx][select,None]
                                    metrics.append(dict(contract=contract,deployable=contract!='P_oracle',oracle_label='ORACLE_NOT_DEPLOYABLE' if contract=='P_oracle' else '',seed=seed,trained_target=target,evaluation_target=evaltarget,scope=scope,log_id=log,signal=({'q_dot':'delta_q_2','az_n':'delta_vz_2'}.get(signal,signal) if evaltarget=='increment2_state' else signal),n=int(select.sum()),rmse=float(np.sqrt(np.mean(np.sum(error**2,axis=1))))))
                folder=a.artifacts/f'{contract}_{target}_seed{seed}';folder.mkdir();torch.save(dict(state_dict=model.cpu().state_dict(),mean=mean,std=std,target_mean=ym,target_std=ys,contract=contract),folder/'model.pt');np.savez_compressed(folder/'predictions.npz',**preds)
                training.append(dict(contract=contract,target=target,seed=seed,epochs=30,steps=30*int(np.ceil(len(fit)/512)),wall_time_s=time.monotonic()-started,checkpoint_sha256=sha(folder/'model.pt')))
                pd.DataFrame(metrics).to_csv(out/'teacher_metrics.csv',index=False);pd.DataFrame(training).to_csv(out/'probe_training.csv',index=False);print('PROBE',target,seed,contract,'complete',flush=True)
    f=pd.DataFrame(metrics);v=f[(f.scope=='validation')&(f.trained_target=='increment2')&(f.evaluation_target=='increment2_state')&f.signal.isin(['angular','linear'])].groupby(['contract','seed','signal']).rmse.mean().unstack('signal')
    agg=v.groupby('contract').mean();benefit=1-agg.loc['P_oracle']/agg.loc['P0'];causal=1-agg.loc['P2']/agg.loc['P0']
    oracle_pass=bool(benefit.angular>=.05 and benefit.linear>=-.05 and ((v.loc['P_oracle'].angular/v.loc['P0'].angular)<1).all())
    causal_pass=bool(causal.angular>=.5*benefit.angular and causal.linear>=-.05)
    gate=dict(oracle_relative_gain=benefit.to_dict(),causal_relative_gain=causal.to_dict(),oracle_pass=oracle_pass,causal_pass=causal_pass,full_training_authorized_by_gate=oracle_pass and causal_pass,rule=protocol['probe_gate'],scope='local diagnostic probe; NOT full simulator upper bound',seeds=protocol['seeds'])
    (out/'probe_gate.json').write_text(json.dumps(gate,indent=2));print(json.dumps(gate,indent=2),flush=True)

if __name__=='__main__':main()
