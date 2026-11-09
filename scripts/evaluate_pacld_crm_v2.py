"""Fixed final-update evaluation. Validation opens only behind the all-arms-64 gate."""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from pacld_crm_v2 import *
from pacld_crm_v1 import A2
from system_identification.data.pacld import open_windows
from system_identification.evaluation.pacld_gpu import predict
from system_identification.evaluation.pahbd import native_horizon_errors
from evaluate_pacld_a2_v2_gpu import batches
from diagnose_pacld_a2 import fit, transform
from diagnose_pacld_a2_v2 import zoh_mean


def require_complete():
    assert json.loads((OUT/'loss_calibration.json').read_text())['status']=='PASS'
    for arm in ['B0','B1','B2']:
        p=LOCAL/arm/'model_64.pt';assert p.exists(),f'{arm} incomplete; validation is forbidden'
        b=torch.load(p,map_location='cpu',weights_only=False)
        assert b['completed_updates']==64 and b['sampler_position']==32768
        assert b['sampling_sha256']==sha(OUT/'shared_sampling_indices.csv')

def factual(model,bundle,frame,split,directory):
    dest=directory/f'pacld_{split}.csv'
    if dest.exists():
        result=pd.read_csv(dest)
        assert len(result)==len(frame) and not result.duplicated(['flight','t']).any()
        assert set(map(tuple,result[['flight','t']].to_numpy()))==set(map(tuple,frame[['flight','t']].to_numpy()))
        return result
    rows=[]
    for cohort,ws,queries in batches(frame,bundle['config'],split,128,4):
        with torch.inference_mode():out=predict(model,ws,queries,base=False)
        for r,w,rr in zip(cohort.itertuples(),ws,out):
            b=rr['belief'].x.numpy();p=rr['predictions']['FULL'][0].x.numpy()
            rows.append(dict(flight=r.flight,date=r.date,t=int(r.t),window_id=w.metadata['window_id'],query_s=rr['times'][0],delta_p=float(p[10]-b[10]),delta_r=float(p[12]-b[12]),belief_p=float(b[10]),belief_r=float(b[12])))
        print('factual',directory.name,split,len(rows),'/',len(frame),flush=True)
    result=pd.DataFrame(rows);result.to_csv(dest,index=False);return result

def features(frame,source):
    raw=np.stack(frame.X);context=np.load(source/'context.npy')
    return np.c_[raw,context,np.stack([frame['mode'].to_numpy()==m for m in [3,15]],axis=1)]

def future_commands(frame,root,split):
    N=np.full((len(frame),8),np.nan)
    for fid,g in frame.groupby('flight',sort=False):
        motor=pd.read_parquet(root/f'native_actuator_motors_{split}.parquet',filters=[('flight_id','==',fid)]).sort_values('timestamp_us')
        servo=pd.read_parquet(root/f'native_actuator_servos_{split}.parquet',filters=[('flight_id','==',fid)]).sort_values('timestamp_us')
        status=pd.read_parquet(root/f'native_vehicle_status_{split}.parquet',filters=[('flight_id','==',fid)]).sort_values('timestamp_us')
        for i,row in g.iterrows():
            t=int(row.t);outputs=[]
            try:
                for tab,fields in [(motor,['control[0]']),(servo,[f'control[{k}]' for k in range(3)])]:
                    ts=tab.timestamp_us.to_numpy();v=tab[fields].to_numpy(float);j=np.searchsorted(ts,t,side='left')-1
                    if j<0 or t-ts[j]>160000:raise ValueError('stale')
                    outputs.append(np.r_[zoh_mean(ts,v,t+250000,t+300000)-v[j],zoh_mean(ts,v,t+250000,t+350000)-v[j]])
                if (status[(status.timestamp_us>t)&(status.timestamp_us<=t+350000)].nav_state!=row['mode']).any():continue
                raw=np.r_[outputs[0][:1],outputs[1][:3],outputs[0][1:],outputs[1][3:]];N[i]=np.r_[transform(raw[:4]),transform(raw[4:])]
            except ValueError:pass
    return N

def moment_rows(arm,split,frame,pred,residuals,keep,scales):
    p=frame[['flight','date','t']].merge(pred,on=['flight','date','t'],validate='one_to_one')
    y=np.stack(frame.Y)[:,0];err=y-p.delta_p.to_numpy();rows=[]
    for j,name in enumerate(['normal','future100']):
        m=residuals[:,j]/scales['s_u'][[0,2][j]]*err/scales['s_p']
        tab=frame[['flight','date']].copy();tab['m']=m;tab=tab.loc[keep]
        flight=tab.groupby(['date','flight']).m.mean();date=flight.groupby('date').mean()
        rng=np.random.default_rng(17);boot=date.to_numpy()[rng.integers(0,len(date),(10000,len(date)))].mean(1)
        for d,v in date.items():rows.append(dict(group=arm,split=split,control=name,date=d,M=v,M2=v*v,dates=1,windows=int((tab.date==d).sum()),ci_low=None,ci_high=None))
        rows.append(dict(group=arm,split=split,control=name,date='ALL',M=date.mean(),M2=date.mean()**2,dates=len(date),windows=len(tab),ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975)))
    return rows

def state_rows(arm,model,bundle,val,frame,pred):
    ids=bundle['data_identity']['validation_window_ids'];lookup={wid:i for i,wid in enumerate(val.index.window_id)};assert len(ids)==156 and set(ids)<=set(lookup)
    rows=[]
    for begin in range(0,len(ids),8):
        ws=[val[lookup[k]] for k in ids[begin:begin+8]]
        with torch.inference_mode():rr=predict(model,ws,base=False)
        for w,r in zip(ws,rr):
            for row in native_horizon_errors(r['predictions']['FULL'],r['times'],w.targets,w.metadata):
                row['window_horizon_us']=w.metadata['horizon_us'];row['date']=val.dataset.provenance[row['flight_id']]['flight_date'];rows.append(row)
    f=pd.DataFrame(rows);f.to_csv(LOCAL/'evaluation'/arm/'native_physical_windows.csv',index=False);output=[]
    for (component,horizon),g in f.groupby(['component','horizon_s']):
        if horizon>1:continue
        for d in ['ALL']+sorted(g.date.unique()):
            gg=g if d=='ALL' else g[g.date==d]
            mse=gg.groupby(['flight_id','window_horizon_us']).squared_error.mean().groupby('flight_id').mean().mean()
            output.append(dict(group=arm,split='validation',cohort='original_fixed156',component=component,horizon_s=horizon,date=d,rmse=math.sqrt(mse)*(180/math.pi if component=='R' else 1),unit={'p':'m','v':'m/s','R':'deg','omega':'rad/s'}[component],dates=gg.date.nunique(),windows=gg.window_id.nunique()))
    p=frame[['flight','date','t']].merge(pred,on=['flight','date','t'],validate='one_to_one');raw=np.stack(frame.X);y=np.stack(frame.Y)
    for axis,idx in [('p',0),('r',2)]:
        target=raw[:,3+idx]+y[:,idx];hat=p['belief_'+axis]+p['delta_'+axis];p['sq']=(target-hat)**2
        for d in ['ALL']+sorted(p.date.unique()):
            g=p if d=='ALL' else p[p.date==d];v=g.groupby(['date','flight']).sq.mean().groupby('date').mean().mean()
            output.append(dict(group=arm,split='validation',cohort='A2_frozen_all_identities',component='omega_'+axis,horizon_s=.2,date=d,rmse=math.sqrt(v),unit='rad/s',dates=g.date.nunique(),windows=len(g)))
    return output

def cir_rows(arm,model,val):
    from pacld_cir_v1 import simulate, TIMES, feasible
    cir=BASE/'controlled_response_v1';selection=pd.read_csv(cir/'selected_validation.csv');protocol=json.loads((cir/'protocol.json').read_text());assert len(selection)==306
    lookup={wid:i for i,wid in enumerate(val.index.window_id)};assert set(selection.window_id)<=set(lookup)
    rows=[]
    for start in range(0,len(selection),64):
        group=selection.iloc[start:start+64];ws=[val[lookup[wid]] for wid in group.window_id]
        ba=compile_windows(ws,model.config).to('cuda:0',torch.float64)
        with torch.inference_mode():belief=model.encode_batch(ba)
        for mult in protocol['multipliers']:
            amplitude=mult*protocol['amplitude'];good=[i for i,w in enumerate(ws) if feasible(w.controls,amplitude)[0]]
            s=belief.with_(**{k:getattr(belief,k)[good] for k in FIELDS});snapshots,_=simulate(model,s,[ws[i].controls for i in good],amplitude)
            for k,t in enumerate(TIMES):
                for j,i in enumerate(good):
                    r=group.iloc[i];p=snapshots[k]['x'][:,j,10]
                    rows.append(dict(group=arm,date=r.date,flight=r.flight,window_id=r.window_id,phase_bin=int(r.phase_bin),multiplier=mult,time=t,gain=float((p[1]-p[2])/(2*amplitude)),positive=float(p[1]-p[0]),negative=float(p[2]-p[0])))
    f=pd.DataFrame(rows);f.to_csv(LOCAL/'evaluation'/arm/'cir_windows.csv',index=False);output=[]
    for (mult,t),g in f.groupby(['multiplier','time']):
        for phase in ['ALL',0,1,2,3]:
            gg=g if phase=='ALL' else g[g.phase_bin==phase]
            output.append(dict(group=arm,multiplier=mult,time=t,phase=str(phase),gain_median=gg.gain.median(),gain_q10=gg.gain.quantile(.1),gain_q90=gg.gain.quantile(.9),positive_median=gg.positive.median(),negative_median=gg.negative.median(),plus_positive_fraction=(gg.positive>0).mean(),minus_negative_fraction=(gg.negative<0).mean(),windows=len(gg),dates=gg.date.nunique()))
    return output

def a2_rows(arm,directory):
    # Reuse original train-nuisance/localization and authorized descriptive validation regression verbatim.
    import score_pacld_a2_v2 as score
    import validation_gain_statistics_a2_v2 as gains
    score.OUT=directory;gains.OUT=directory
    score.main();gains.main()
    g=pd.read_csv(directory/'validation_evaluation_gain_summary.csv')
    g=g[(g.axis=='p')&(g.channel=='differential')].copy();g['group']=arm
    s=pd.read_csv(directory/'validation_summary.csv');s=s[(s.axis=='p')&(s.model=='PACLD')].iloc[0]
    g['independent_response_score']=s.response_score;g['score_low']=s.low;g['score_high']=s.high;g['dates']=s.dates
    return g

def main():
    require_complete();seed()
    if (LOCAL/'evaluation_status.json').exists() and json.loads((LOCAL/'evaluation_status.json').read_text())['status']=='complete':return
    write_json(LOCAL/'evaluation_status.json',dict(status='evaluating',validation_authorized_after_all64=True))
    original=torch.load(CHECKPOINT,map_location='cpu',weights_only=False);scales=json.loads((V1/'moments.json').read_text())['scales']
    train=pd.read_pickle(SOURCE/'features.pkl').reset_index(drop=True);vsource=A2/'validation_features';validation=pd.read_pickle(vsource/'features.pkl').reset_index(drop=True)
    X=features(train,SOURCE);V=features(validation,vsource);U=np.stack(train.U);W=np.stack(validation.U)
    a=np.load(A2/'features.npz');np.testing.assert_allclose(X,a['X']);np.testing.assert_allclose(U,a['U'])
    normal=fit(X,U,train.date.to_numpy());future=fit(X,a['N'],train.date.to_numpy())
    N=future_commands(validation,Path(original['config']['data']['root']),'validation')
    keep=np.isfinite(N).all(1)&validation['mode'].isin([3,15]).to_numpy()
    nv=np.zeros((len(validation),2));nv[:,0]=(W-normal.predict(V))[:,5];nv[:,1]=(N-future.predict(V))[:,5]
    tmeta=pd.read_csv(V1/'windows.csv');joined=train[['flight','t']].merge(tmeta,on=['flight','t'],validate='one_to_one');nt=joined[['R_normal','R_future100']].to_numpy()
    val=open_windows(original['config'],'validation')
    for key in ['manifest_sha256','index_sha256','windows','split']:assert val.identity()[key]==original['data_identity']['validation'][key]
    moment=[];state=[];cir=[];a2=[]
    for arm in ['initial','B0','B1','B2']:
        bundle=original if arm=='initial' else torch.load(LOCAL/arm/'model_64.pt',map_location='cpu',weights_only=False)
        directory=LOCAL/'evaluation'/arm;directory.mkdir(parents=True,exist_ok=True)
        identity=dict(model_sha256=SHA if arm=='initial' else sha(LOCAL/arm/'model_64.pt'),evaluation_source_sha256=sha(__file__),baseline=BASE_SHA)
        ip=directory/'identity.json'
        if ip.exists():assert json.loads(ip.read_text())==identity
        else:write_json(ip,identity)
        link=directory/'validation_features'
        if not link.exists():link.symlink_to(vsource.resolve(),target_is_directory=True)
        model,_=model_and_optimizer(bundle);model.eval()
        pt=factual(model,bundle,train,'train',directory);pv=factual(model,bundle,validation,'validation',directory)
        moment+=moment_rows(arm,'train',train,pt,nt,np.ones(len(train),bool),scales)
        moment+=moment_rows(arm,'validation',validation,pv,nv,keep,scales)
        state+=state_rows(arm,model,original,val,validation,pv);cir+=cir_rows(arm,model,val)
        a2.append(a2_rows(arm,directory));del model;torch.cuda.empty_cache()
        pd.DataFrame(moment).to_csv(OUT/'moment_comparison.csv',index=False);pd.DataFrame(state).to_csv(OUT/'state_prediction_comparison.csv',index=False);pd.DataFrame(cir).to_csv(OUT/'cir_comparison.csv',index=False);pd.concat(a2).to_csv(OUT/'a2_comparison.csv',index=False)
    write_json(LOCAL/'evaluation_status.json',dict(status='complete',validation_dates=int(validation.date.nunique()),moment_valid_windows=int(keep.sum()),excluded_moment_windows=int((~keep).sum()),selection_depends_on_model=False,sealed_test_opened=False))

if __name__=='__main__':main()
