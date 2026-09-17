#!/usr/bin/env python3
"""Supplementary target/phase/telemetry diagnostics after the primary audit."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-main-v2-step4')
import sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd,torch
from pyulog import ULog
from main_v2_step4_tools import harmonic_design,vector_r2,nearest_indices
from run_main_v2_step4 import load_segments,csv,describe
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);pa.add_argument('--device',default='cuda:1');args=pa.parse_args();out=args.output
    assert json.loads((out/'manifest.json').read_text())['stage']=='analysis_completed'
    torch.set_num_threads(4)
    allmeta={s:pd.read_csv(out/(s+'_points.csv')) for s in ['train','validation']}
    arrays={s:np.load(out/(s+'_diagnostic_arrays.npz')) for s in allmeta}
    phase=[];statistics=[];ext={};sensor_compare=[];inventory=[]
    for split,m in allmeta.items():
        a=arrays[split];z=a['target_D0_raw'];p=a['prediction'];additional=np.zeros((len(m),4))*np.nan
        for log,g in m.groupby('log_id'):
            ix=g.index.to_numpy();X=harmonic_design(g.phase.to_numpy(),3);cut=g.timestamp_s.median();fit=g.timestamp_s.to_numpy()<cut-1;test=g.timestamp_s.to_numpy()>cut+1
            # Both fits remain descriptive within-log diagnostics, including validation.
            cy=np.linalg.lstsq(X[fit],z[ix][fit],rcond=None)[0];cp=np.linalg.lstsq(X[fit],p[ix][fit],rcond=None)[0]
            for sig,sl in [('linear',slice(0,3)),('angular',slice(3,6))]:
                component=(X@cy)[:,sl];pc=(X@cp)[:,sl]
                phase.append(dict(split=split,log_id=log,signal=sig,component_heldout_r2=vector_r2(component[test],z[ix][test,sl]),model_phase_component_rmse=float(np.sqrt(np.mean(np.sum((pc[test]-component[test])**2,axis=1)))),component_truth_std=float(np.linalg.norm(component[test].std(0))),component_pred_std=float(np.linalg.norm(pc[test].std(0))),prediction_vs_component_rmse=float(np.sqrt(np.mean(np.sum((p[ix][test,sl]-component[test])**2,axis=1))))))
            # Strict past-only telemetry alignment. Battery current is pack current, not motor torque.
            u=ULog(str(Path('/home/zn/QgcLogs')/log))
            for d in u.data_list:
                inventory.append(dict(split=split,log_id=log,topic=d.name,instance=d.multi_id,count=len(d.data['timestamp'])))
                if d.multi_id!=0:continue
                time=np.asarray(d.data['timestamp'])*1e-6;j=np.searchsorted(time,g.timestamp_s,side='right')-1;safe=np.clip(j,0,len(time)-1);valid=(j>=0)&((g.timestamp_s.to_numpy()-time[safe])<.25)
                if d.name in ['battery_status','rpm']:
                    for field,col in ([('current_a',0),('voltage_v',1)] if d.name=='battery_status' else [('rpm_raw',2),('rpm_estimate',3)]):
                        val=np.asarray(d.data[field],float)[safe];val[~valid]=np.nan;additional[ix,col]=val
                if d.name=='vehicle_angular_velocity':
                    # Published derivative is filtered sensor-derived output, not independent physical truth.
                    zd=np.column_stack([d.data[f'xyz_derivative[{k}]'][safe] for k in range(3)])
                    zr=np.column_stack([d.data[f'xyz[{k}]'][safe] for k in range(3)])
                    for sig,v in [('published_angular_derivative',zd),('angular_velocity',zr)]:
                        for k in range(3):sensor_compare.append(dict(split=split,log_id=log,signal=sig,axis=k,**describe(v[valid,k])))
                    if np.isfinite(zd[valid]).all():sensor_compare.append(dict(split=split,log_id=log,signal='published_vs_D0',axis='vector',rmse=float(np.sqrt(np.mean(np.sum((zd[valid]-z[ix][valid,3:])**2,axis=1))))))
        ext[split]=additional
        # Exact aggregate moments distinguish one-step teacher statistics from recursive 3.30.
        for target in ['D0_raw','D1_lp4','D1_lp6','D1_lp8','D1_lp12','increment2','increment5','increment10']:
            for sig,sl in [('linear',slice(0,3)),('angular',slice(3,6))]:
                y=a['target_'+target][:,sl];pred=p[:,sl]
                statistics.append(dict(split=split,target=target,signal=sig,truth_std=float(np.linalg.norm(y.std(0))),pred_std=float(np.linalg.norm(pred.std(0))),rmse=float(np.sqrt(np.mean(np.sum((pred-y)**2,axis=1)))),r2=vector_r2(pred,y)))
        np.savez_compressed(out/(split+'_telemetry_arrays.npz'),values=additional,columns=np.array(['battery_current_a','battery_voltage_v','rpm_raw','rpm_estimate']))
    csv(out,'teacher_phase_component',phase);csv(out,'teacher_aggregate',statistics);csv(out,'published_sensor_derivative',sensor_compare);csv(out,'raw_topic_inventory',inventory)
    # Extra-channel information value: kNN under same fixed identities/time exclusions.
    rows=[];refm=allmeta['train'];ri=np.arange(0,len(refm),5);refa=arrays['train']
    for split,m in allmeta.items():
        qidx=np.concatenate([g.index.to_numpy()[np.linspace(0,len(g)-1,min(200,len(g)),dtype=int)] for _,g in m.groupby('log_id')]);qm=m.iloc[qidx];rm=refm.iloc[ri]
        same=qm.log_id.to_numpy()[:,None]==rm.log_id.to_numpy()[None];close=np.abs(qm.timestamp_s.to_numpy()[:,None]-rm.timestamp_s.to_numpy()[None])<2
        for scope in (['same_log','other_log'] if split=='train' else ['train_only']):
            forbidden=(~same if scope=='same_log' else same if scope=='other_log' else np.zeros_like(same))|(same&close)
            for rep,cols in [('plus_pack_voltage_current',[0,1]),('plus_raw_rpm',[2]),('plus_all_load_channels',[0,1,2,3])]:
                x=np.column_stack([refa['feature_gru26'][ri],ext['train'][ri][:,cols]]);q=np.column_stack([arrays[split]['feature_gru26'][qidx],ext[split][qidx][:,cols]])
                mean=np.nanmean(x,0);std=np.nanstd(x,0);std[std<1e-6]=1;x=np.nan_to_num((x-mean)/std);q=np.nan_to_num((q-mean)/std)
                nn,dist=nearest_indices(q,x,forbidden,k=20,device=args.device)
                for target in ['D0_raw','D1_lp6','increment2']:
                    ty=refa['target_'+target][ri];qy=arrays[split]['target_'+target][qidx];pr=ty[nn].mean(1);cv=ty[nn].var(1,ddof=1)
                    for log,g in qm.reset_index(drop=True).groupby('log_id'):
                        j=g.index.to_numpy()
                        for sig,sl in [('linear',slice(0,3)),('angular',slice(3,6))]:rows.append(dict(split=split,scope=scope,log_id=log,representation=rep,target=target,signal=sig,k=20,conditional_variance_ratio=float(cv[j,sl].sum(1).mean()/ty[:,sl].var(0).sum()),knn_rmse=float(np.sqrt(np.mean(np.sum((pr[j,sl]-qy[j,sl])**2,axis=1)))),distance_median=float(np.median(dist[j]))))
    csv(out,'load_channel_information',rows)
    # Conditioning-distance strata expose whether neighbors are actually close.
    n=pd.read_csv(out/'neighbor_ids.csv');close_rows=[]
    for (split,scope,rep,k),g in n.groupby(['split','scope','representation','k']):
        j=g.query_index.to_numpy();nn=np.array([np.fromstring(s,sep=' ',dtype=int) for s in g.neighbor_indices]);dist=g.distance_rms.to_numpy()
        for name in ['D0_raw','D1_lp6','increment2']:
            ty=refa['target_'+name];qy=arrays[split]['target_'+name][j];v=ty[nn].var(1,ddof=1)
            for radius in [.25,.5,1.]:
                sel=dist<=radius
                for sig,sl in [('linear',slice(0,3)),('angular',slice(3,6))]:close_rows.append(dict(split=split,scope=scope,representation=rep,k=k,target=name,signal=sig,radius=radius,n_queries=int(sel.sum()),conditional_variance_ratio=float(v[sel,sl].sum(1).mean()/ty[:,sl].var(0).sum()) if sel.any() else np.nan))
    csv(out,'conditional_distance_strata',close_rows)
    # Sensitivity distributions and t0 control bins fixed by training quantiles.
    s=pd.read_csv(out/'command_sensitivity.csv');sens=[]
    for split in ['train','validation']:
        ix=s.index[s.split==split];point=s.loc[ix,'point_index'].to_numpy(dtype=int);features=arrays[split]['feature_instant'][point]
        s.loc[ix,'speed']=np.linalg.norm(features[:,:3],axis=1)
        s.loc[ix,'body_rate_norm']=np.linalg.norm(features[:,3:6],axis=1)
        s.loc[ix,'frequency']=features[:,11]
    for variable in ['motor','tail_magnitude','speed','body_rate_norm','frequency']:
        edges=np.quantile(s[s.split=='train'][variable],[1/3,2/3]);s['bin']=np.searchsorted(edges,s[variable],side='right')
        for keys,g in s.groupby(['split','channel','step','output','bin']):
            a=g.sensitivity.to_numpy();sens.append(dict(variable=variable,split=keys[0],channel=keys[1],step=keys[2],output=keys[3],bin=keys[4],bin_edges=json.dumps(edges.tolist()),near_zero_fraction=float(np.mean(abs(a)<1e-6)),positive_fraction=float(np.mean(a>1e-6)),negative_fraction=float(np.mean(a< -1e-6)),**describe(a)))
    csv(out,'sensitivity_bins',sens)
    # Relationship to tail commands and frequency/battery bins, not control causation.
    assoc=[]
    tr=arrays['train']['feature_instant'];edges_tail=np.quantile(np.linalg.norm(tr[:,13:16],axis=1),[1/3,2/3]);edges_f=np.quantile(allmeta['train'].frequency,[1/3,2/3])
    for split,m in allmeta.items():
        a=arrays[split];tail=np.linalg.norm(a['feature_instant'][:,13:16],axis=1);tb=np.searchsorted(edges_tail,tail);fb=np.searchsorted(edges_f,m.frequency)
        err=np.linalg.norm(a['prediction'][:,3:]-a['target_D0_raw'][:,3:],axis=1)
        for log,g in m.groupby('log_id'):
            for tbin in range(3):
                ix=g.index.to_numpy();ix=ix[tb[ix]==tbin]
                if len(ix):assoc.append(dict(split=split,log_id=log,kind='tail',bin=tbin,n=len(ix),angular_derivative_error_mean=float(err[ix].mean())))
            for fbin in range(3):
                ix=g.index.to_numpy();ix=ix[fb[ix]==fbin]
                if len(ix)>20:
                    v=ext[split][ix,0];valid=np.isfinite(v)
                    assoc.append(dict(split=split,log_id=log,kind='frequency_conditioned_pack_current',bin=fbin,n=len(ix),angular_derivative_error_mean=float(err[ix].mean()),current_error_correlation=float(np.corrcoef(v[valid],err[ix][valid])[0,1]) if v[valid].std()>1e-8 else np.nan))
    csv(out,'actuator_associations',assoc)
    # Visual comparison of raw/filtered/increment teacher-state target fidelity.
    f=pd.DataFrame(statistics);fig,axes=plt.subplots(1,2,figsize=(13,4))
    for ax,signal in zip(axes,['linear','angular']):
        for split in ['train','validation']:
            g=f[(f.split==split)&(f.signal==signal)];ax.plot(g.target,g.truth_std,'o-',label=split+' target');ax.axhline(g.pred_std.iloc[0],linestyle='--',label=split+' teacher prediction')
        ax.tick_params(axis='x',rotation=60);ax.set(title=signal,ylabel='Vector std');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'target_variance.png',dpi=160);plt.close(fig)
    (out/'supplement_complete.json').write_text(json.dumps(dict(completed=True,sealed_test_opened=False),indent=2))

if __name__=='__main__':main()
