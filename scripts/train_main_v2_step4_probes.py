#!/usr/bin/env python3
"""Small matched diagnostic regressors; never replaces Main V2 or its simulator."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json,argparse,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd,torch
from main_v2_step4_tools import vector_r2
from run_main_v2_free_running import sha


def phase_history(x,phi):
    x=x.copy();h=x[:,:26*12].reshape(-1,26,12);s=h[:,:,9].copy();c=h[:,:,10].copy()
    h[:,:,9]=s*np.cos(phi[:,None])+c*np.sin(phi[:,None]);h[:,:,10]=c*np.cos(phi[:,None])-s*np.sin(phi[:,None]);return x


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--results',type=Path,required=True);pa.add_argument('--artifacts',type=Path,required=True);pa.add_argument('--device',default='cuda:1');args=pa.parse_args()
    out=args.results
    assert (out/'supplement_complete.json').exists(),'complete phase/target/observability audit before probes'
    if args.artifacts.exists() and any(args.artifacts.iterdir()):raise FileExistsError(args.artifacts)
    args.artifacts.mkdir(parents=True)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    m={s:pd.read_csv(out/(s+'_points.csv')) for s in ['train','validation']};a={s:np.load(out/(s+'_diagnostic_arrays.npz')) for s in m}
    offsets=pd.read_csv(out/'train_phase_offsets.csv').set_index('log_id').phase_offset_rad.to_dict()
    # np.roll(curve,k) alignment means new phase = original phase + k*bin_width.
    raw={s:a[s]['feature_history26'].astype(np.float32) for s in m};corrected={s:phase_history(raw[s],m[s].phase.to_numpy()+np.array([offsets.get(log,0.) for log in m[s].log_id])) for s in m}
    fit=np.zeros(len(m['train']),bool);hold=fit.copy()
    for log,g in m['train'].groupby('log_id'):
        cut=float(np.quantile(g.timestamp_s,.7));fit[g.index]=g.timestamp_s<cut-2;hold[g.index]=g.timestamp_s>cut+2
    # Deterministic downsampling for a small budget, same rows for all models.
    fitix=np.flatnonzero(fit)[::2];holdix=np.flatnonzero(hold);validix=np.arange(len(m['validation']))
    mean=raw['train'][fitix].mean(0);std=raw['train'][fitix].std(0);std[std<1e-5]=1
    phasecols=np.array([12*i+j for i in range(26) for j in [9,10]]);mean[phasecols]=0;std[phasecols]=1
    ym=a['train']['target_D0_raw'][fitix].mean(0);ys=a['train']['target_D0_raw'][fitix].std(0)
    protocol=dict(purpose='diagnostic only; not Main V3',architecture='MLP input420 ->64 SiLU ->64 SiLU ->6',input='26-point raw causal history plus commands and actuator proxies; no pretrained GRU to contaminate temporal holdout',seeds=[415,416,417],epochs=30,batch_size=512,optimizer='AdamW lr0.001 weight_decay0.00001',target_normalization='same raw-train scales in every cell',phase_normalization='fixed sin/cos mean0 std1 for both representations',phase_correction='train earlier-half constant offsets; validation offset unknown and set0; no validation offset calibration',fit_scope='first70percent each train log minus2s; temporal heldout last30percent plus2s; fit_stride2',n_train=len(fitix),n_temporal_holdout=len(holdix),n_validation=len(validix),filter_use='no filtered state as input',cells=dict(B0=['current_phase','D0_raw'],B1=['current_phase','increment2'],B2=['log_phase_train_offset','D0_raw'],B3=['log_phase_train_offset','increment2']),sealed_test_opened=False)
    (out/'probe_protocol.json').write_text(json.dumps(protocol,indent=2))
    pd.DataFrame(dict(point_index=fitix,role='fit')).to_csv(out/'probe_fit_ids.csv',index=False);pd.DataFrame(dict(point_index=holdix,role='train_temporal_holdout')).to_csv(out/'probe_holdout_ids.csv',index=False)
    metrics=[];training=[]
    for seed in protocol['seeds']:
        for name,(representation,target) in protocol['cells'].items():
            started=time.monotonic();torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            x=raw if representation=='current_phase' else corrected
            model=torch.nn.Sequential(torch.nn.Linear(raw['train'].shape[1],64),torch.nn.SiLU(),torch.nn.Linear(64,64),torch.nn.SiLU(),torch.nn.Linear(64,6)).to(args.device)
            opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-5)
            xt=torch.tensor((x['train'][fitix]-mean)/std,device=args.device,dtype=torch.float32);yt=torch.tensor((a['train']['target_'+target][fitix]-ym)/ys,device=args.device,dtype=torch.float32)
            gen=torch.Generator().manual_seed(seed);torch.cuda.reset_peak_memory_stats(args.device)
            for epoch in range(30):
                ix=torch.randperm(len(fitix),generator=gen)
                for start in range(0,len(ix),512):
                    j=ix[start:start+512].to(args.device);opt.zero_grad(set_to_none=True);loss=(model(xt[j])-yt[j]).square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            model.eval();predictions={}
            with torch.inference_mode():
                for scope,split,ix in [('fit','train',fitix),('train_temporal_holdout','train',holdix),('validation_unknown_offset','validation',validix)]:
                    pred=np.concatenate([model(torch.tensor((x[split][j]-mean)/std,device=args.device,dtype=torch.float32)).cpu().numpy() for j in np.array_split(ix,max(1,int(np.ceil(len(ix)/4096))))])*ys+ym
                    predictions[scope]=pred
                    for evaltarget in ['D0_raw','D1_lp12','increment2','increment5','increment10']:
                        y=a[split]['target_'+evaltarget][ix]
                        for log in m[split].iloc[ix].log_id.unique():
                            select=m[split].iloc[ix].log_id.to_numpy()==log
                            for signal,sl in [('linear',slice(0,3)),('angular',slice(3,6))]:
                                p=pred[select,sl];truth=y[select,sl];metrics.append(dict(model=name,seed=seed,scope=scope,log_id=log,trained_target=target,evaluation_target=evaltarget,signal=signal,n=len(p),rmse=float(np.sqrt(np.mean(np.sum((p-truth)**2,axis=1)))),r2=vector_r2(p,truth),pred_std=float(np.linalg.norm(p.std(0))),truth_std=float(np.linalg.norm(truth.std(0)))))
            folder=args.artifacts/f'{name}_seed{seed}';folder.mkdir();torch.save(dict(state_dict=model.cpu().state_dict(),feature_mean=mean,feature_std=std,target_mean=ym,target_std=ys,protocol=protocol),folder/'model.pt');np.savez_compressed(folder/'predictions.npz',**predictions)
            training.append(dict(model=name,seed=seed,epochs=30,optimizer_steps=30*int(np.ceil(len(fitix)/512)),wall_time_s=time.monotonic()-started,gpu_peak_bytes=torch.cuda.max_memory_allocated(args.device),checkpoint_sha256=sha(folder/'model.pt'),final_batch_loss=float(loss)))
            pd.DataFrame(metrics).to_csv(out/'probe_metrics.csv',index=False);pd.DataFrame(training).to_csv(out/'probe_training.csv',index=False)
            print('probe done',name,seed,flush=True)
    (out/'probe_complete.json').write_text(json.dumps(dict(completed=True,sealed_test_opened=False),indent=2))

if __name__=='__main__':main()
