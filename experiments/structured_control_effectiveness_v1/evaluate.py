"""Frozen pair identities and A/B/D references; exact original receiver/donor tapes."""
from common import *
import torch
from model import make_model
from system_identification.models.velocity_rate_gru import errors
tc=module(ROOT/'experiments/local_control_response_timing_slowfast_v1/common.py','timing')
tm=legacy(ROOT/'experiments/local_control_response_timing_slowfast_v1/analyze.py','timing_metric')
stats=module(ROOT/'experiments/local_control_response_rollout_v1/analyze.py','dependency_weights')
sm=legacy(SWAP/'analyze.py','swap_metric')
VARS=['vx','vy','vz','p','q','r'];CHANNELS=[3,4,5,10,11,12]

def predict():
    cfg=json.loads((HERE/'config.json').read_text());device=torch.device(cfg['device'])
    torch.set_num_threads(4);torch.cuda.set_device(device);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    x=np.load(SWAP/'results/inputs.npz');inputs={k:torch.from_numpy(x[k]).to(device) for k in training.INPUT_KEYS};pred={};checks=[]
    with torch.inference_mode():
        for name in cfg['models']:
            path=HERE/'training'/name/'best.pt';ck=torch.load(path,map_location=device,weights_only=False)
            m=make_model(name,ck['normalization']).to(device);m.load_state_dict(ck['model']);m.eval()
            p=m(inputs);pred[name]=p.cpu().numpy()
            one=torch.cat([m({k:v[j:j+1] for k,v in inputs.items()}) for j in range(4)])
            gap=float(abs(p[:4]-one).max());assert gap<2e-5,gap
            poison=dict(inputs,target=torch.full_like(p,float('nan')),future_phase=torch.full_like(inputs['dt'],float('nan')))
            assert torch.equal(p,m(poison))
            checks.append(dict(model=name,checkpoint_sha256=sha(path),epoch=ck['epoch'],batch_single_max_error=gap,future_truth_excluded=True))
    np.savez_compressed(OUT/'predictions.npz',**pred)
    save(OUT/'inference_checks.json',dict(passed=True,checks=checks,inputs_sha256=sha(SWAP/'results/inputs.npz'),test_read=False))

def build():
    cfg=json.loads((HERE/'config.json').read_text());cases=pd.read_csv(SWAP/'results/cases.csv');pairs=pd.read_csv(SWAP/'results/pairs.csv')
    x=np.load(SWAP/'results/inputs.npz');pred=np.load(OUT/'predictions.npz')
    # Immutable reference tables already contain audited physical-time truth and support.
    truth=pd.read_csv(ABLATION/'results/factual_curves.csv.gz');truth=truth[(truth.variant=='V0')&(truth.model=='GRU-Control')&(truth.seed==17)]
    template=pd.read_csv(ABLATION/'results/internal_curves.csv.gz');template=template[(template.variant=='V0')&(template.model=='GRU-Control')&(template.seed==17)]
    facts=[];effects=[];windows=[]
    for pi,pair in pairs.iterrows():
        original=truth[truth.pair_uid==pair.pair_uid];tt=original[['time_ms','time_kind','time_label']].drop_duplicates();at=tt.time_ms.to_numpy()/1000
        index={(r.time_label,r.variable,r.method):r for r in original.itertuples()}
        internal={(r.time_label,r.variable):r for r in template[template.pair_uid==pair.pair_uid].itertuples()}
        for name in pred.files:
            ys=[];smooth=[]
            for ci in range(pi*4,pi*4+4):
                n=int(x['valid_steps'][ci]);t=x['native_time_us'][ci,:n+1]*1e-6;y=np.vstack([x['initial'][ci],pred[name][ci,:n]]).astype(float);y-=y[0].copy()
                ys.append(tc.interp(t,y[:,CHANNELS],at));smooth.append(tc.boxmean(t,y[:,CHANNELS],at,.27))
                for j,r in enumerate(tt.itertuples()):
                    if r.time_kind!='grid' or round(r.time_ms) in cfg['report_times_ms']:
                        for k,var in enumerate(VARS):windows.append(dict(model=name,seed=17,case_id=ci,pair_uid=pair.pair_uid,split=pair.split,cell=cases.cell.iloc[ci],receiver_window=cases.receiver_window.iloc[ci],donor_window=cases.donor_window.iloc[ci],variable=var,time_label=r.time_label,time_ms=r.time_ms,increment=ys[-1][j,k]))
            aa,ab,ba,bb=ys;ca=aa-ab;cb=ba-bb;ha=aa-ba;hb=ab-bb;c=(ca+cb)/2;h=(ha+hb)/2;f=aa-bb
            for ti,r in enumerate(tt.itertuples()):
                for j,var in enumerate(VARS):
                    base=dict(model=name,seed=17,pair_uid=pair.pair_uid,split=pair.split,variable=var,time_ms=r.time_ms,time_kind=r.time_kind,time_label=r.time_label)
                    old=internal[(r.time_label,var)];row=dict(**base,supported=old.supported,gA=old.gA,gB=old.gB,g=old.g,real_F=old.real_F)
                    for key,val in zip(['YAA','YAB','YBA','YBB','CA','CB','HA','HB','C','H','F','interaction'],[aa,ab,ba,bb,ca,cb,ha,hb,c,h,f,ca-cb]):row[key]=val[ti,j] if old.supported else np.nan
                    effects.append(row)
                    for method in (['A','B','D'] if j>=3 else ['A']):
                        old=index[(r.time_label,var,method)];value=(smooth[0][ti,j]-smooth[3][ti,j]) if method=='D' else f[ti,j]
                        assert np.isfinite(value) or not old.supported
                        facts.append(dict(**base,method=method,supported=old.supported,real=old.real,pred=value if old.supported else np.nan,error=value-old.real if old.supported else np.nan))
    f=pd.DataFrame(facts);e=pd.DataFrame(effects);f.to_csv(OUT/'factual_curves.csv.gz',index=False);e.to_csv(OUT/'internal_curves.csv.gz',index=False);pd.DataFrame(windows).to_csv(OUT/'window_points.csv.gz',index=False)
    selected=lambda d:d[(d.time_kind!='grid')|d.time_ms.round(6).isin(cfg['report_times_ms'])].copy()
    f=selected(f);e=selected(e);f.to_csv(OUT/'factual_points.csv.gz',index=False);e.to_csv(OUT/'internal_points.csv.gz',index=False)
    return f,e

def summarize(f,e):
    pairs=pd.read_csv(SWAP/'results/pairs.csv');mem=pd.read_csv(SWAP/'results/cohorts.csv');summ=[];ins=[];bygroup=[]
    for (split,cohort),members in mem.groupby(['split','cohort']):
        pp=pairs[pairs.pair_uid.isin(members.pair_uid)]
        for frame,internal in [(f,False),(e,True)]:
            selected=frame[frame.pair_uid.isin(pp.pair_uid)&frame.supported];keys=['model','variable','time_label']+([] if internal else ['method']);metric=sm.metrics if internal else tm.metric
            for key,g in selected.groupby(keys,sort=False):
                meta=dict(zip(keys,key));effective=pp[pp.pair_uid.isin(g.pair_uid)]
                for weighting in ['pair','flight','component'] if meta['time_label']=='300' else ['flight']:
                    row=dict(split=split,cohort=cohort,seed=17,**meta,weighting=weighting,frozen_pairs=len(pp),**metric(g,g.pair_uid.map(stats.weights(effective,weighting))));(ins if internal else summ).append(row)
                if cohort=='main' and meta['time_label']=='300' and meta['variable']=='p':
                    graph=stats.labels(pp)
                    for field,values in [('flight',set(pp.flight_a)|set(pp.flight_b)),('component',set(graph.values())),('date',set(pp.date_a)|set(pp.date_b)),('controller',set(pp.controller_a)|set(pp.controller_b))]:
                        for value in sorted(values):
                            ids=[u for u,v in graph.items() if v==value] if field=='component' else pp.loc[(pp[field+'_a']==value)|(pp[field+'_b']==value),'pair_uid'];sub=g[g.pair_uid.isin(ids)]
                            bygroup.append(dict(split=split,kind='internal' if internal else 'factual',**meta,grouping=field,group=value,**metric(sub,np.ones(len(sub)))))
    pd.DataFrame(summ).to_csv(OUT/'factual_summary.csv',index=False);pd.DataFrame(ins).to_csv(OUT/'internal_summary.csv',index=False);pd.DataFrame(bygroup).to_csv(OUT/'per_group.csv',index=False)
    main=pd.DataFrame(summ).query("split=='train' and cohort=='main' and variable=='p' and time_label=='300' and weighting=='flight'")
    main.to_csv(OUT/'main_p_300.csv',index=False);print(main[['model','method','direction_accuracy','scale_ratio','signed_gain','rmse']].to_string(index=False),flush=True)
    keys=['pair_uid','split','seed','variable','method','time_label'];base=f[f.model=='M0'];delta=f[f.model!='M0'].merge(base,on=keys,suffixes=('','_M0'),validate='many_to_one')
    delta['prediction_change']=delta.pred-delta.pred_M0;delta['absolute_error_change']=abs(delta.error)-abs(delta.error_M0);delta.to_csv(OUT/'paired_changes.csv.gz',index=False)

def trajectory():
    cfg=json.loads((HERE/'config.json').read_text());device=torch.device(cfg['device']);vd=load_data('validation',device);meta=pd.read_parquet(TRAIN/'validation_windows.parquet');records=[];sample={};sample_ids=set(np.linspace(0,len(meta)-1,128,dtype=int))
    with torch.inference_mode():
        for name in cfg['models']:
            ck=torch.load(HERE/'training'/name/'best.pt',map_location=device,weights_only=False);m=make_model(name,ck['normalization']).to(device);m.load_state_dict(ck['model']);m.eval()
            for ix in torch.arange(len(meta),device=device).split(2048):
                y=m(training.batch(vd,ix));er=errors(y,vd['target'][ix]);mask=vd['future_mask'][ix];count=mask.sum(1);last=count-1
                mean=((er.square()*mask[...,None]).sum(1)/count[:,None]).sqrt().cpu().numpy();end=er[torch.arange(len(ix),device=device),last].cpu().numpy()
                for k,i in enumerate(ix.cpu().numpy()):
                    if i in sample_ids:
                        sample[f'{name}_{i}']=y[k].cpu().numpy()
                        sample[f'target_{i}']=vd['target'][i].cpu().numpy()
                        sample[f'mask_{i}']=vd['future_mask'][i].cpu().numpy()
                    row=dict(model=name,window_index=int(i),flight_id=meta.flight_id.iloc[i])
                    for j,var in enumerate(['position_m','velocity_mps','attitude_rad','rate_radps']):row[var+'_path_rms']=mean[k,j];row[var+'_endpoint_norm']=end[k,j]
                    records.append(row)
    np.savez_compressed(OUT/'trajectory_verification_samples.npz',**sample)
    d=pd.DataFrame(records);d.to_csv(OUT/'validation_trajectory.csv.gz',index=False);g=d.groupby(['model','flight_id']).mean(numeric_only=True).drop(columns='window_index');g.to_csv(OUT/'validation_trajectory_per_flight.csv');g.groupby('model').mean().to_csv(OUT/'validation_trajectory_summary.csv')
if __name__=='__main__':predict();summarize(*build());trajectory()
