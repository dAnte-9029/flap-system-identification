"""Strict held-flight evaluation; pilot metric/filter functions reused unchanged."""
from common import *
import argparse,torch
from system_identification.models.velocity_rate_gru import errors
sys.path.insert(0,str(PILOT))
try:old=legacy(PILOT/'evaluate.py','pilot_evaluation')
finally:sys.path.remove(str(PILOT))
VARS=old.VARS;CHANNELS=old.CHANNELS

def load_model(fold,name,seed,device):
    folder=HERE/'folds'/f'fold{fold}'/'training'/name/f'seed{seed}'
    record=json.loads((folder/'summary.json').read_text());assert record['epochs']==6
    path=folder/'last.pt';assert sha(path)==record['last_sha256']
    ck=torch.load(path,map_location=device,weights_only=False);assert ck['epoch']==6;norm=json.loads((HERE/'folds'/f'fold{fold}'/'normalization.json').read_text());assert ck['normalization']==norm
    model=make_model(name,norm).to(device);model.load_state_dict(ck['model']);model.eval();return model,record

def predict(device):
    x=np.load(SWAP/'results/inputs.npz');pairs=pd.read_csv(OUT/'pairs.csv');plan=json.loads((OUT/'holdout_protocol.json').read_text());checks=[];support=[]
    refine=module(ROOT/'experiments/control_event_canonicalization_v1/run.py','canonical_inference').refine
    for f in plan['folds']:
        fold=f['fold'];uids=set(f['eligible_all_frozen_pairs'])|{'validation:44'}
        pp=pairs[pairs.pair_uid.isin(uids)];case_ids=np.array([ci for pi in pp.index for ci in range(pi*4,pi*4+4)])
        for p in pp.itertuples():
            if p.split=='train':assert p.flight_a in f['held_flights'] and p.flight_b in f['held_flights']
            support.append(dict(fold=fold,pair_uid=p.pair_uid,role='heldout' if p.split=='train' else 'selection_exposed_case',both_flights_unseen=p.split=='train'))
        data={k:torch.from_numpy(x[k][case_ids]).to(device) for k in training.INPUT_KEYS};result={'case_ids':case_ids}
        with torch.inference_mode():
            for name in COMBOS:
                for seed in [17]:
                    model,record=load_model(fold,name,seed,device);p=model(data);result[f'{name}_{seed}']=p.cpu().numpy()
                    small={k:v[:4] for k,v in data.items()};a=model(small);single=torch.cat([model({k:v[j:j+1] for k,v in small.items()}) for j in range(4)])
                    batch_gap=float(abs(a-single).max());assert batch_gap<2e-5,batch_gap
                    ev,le=refine(small['events'].cpu().numpy(),small['event_length'].cpu().numpy(),4)
                    torch.testing.assert_close(a,model(dict(small,events=torch.from_numpy(ev).to(device),event_length=torch.from_numpy(le).to(device))),rtol=0,atol=0)
                    assert torch.equal(p,model(dict(data,target=torch.full_like(p,float('nan')),future_phase=torch.full_like(data['dt'],float('nan')))))
                    checks.append(dict(fold=fold,model=name,seed=seed,epoch=6,checkpoint_sha256=record['last_sha256'],batch_single_max_error=batch_gap,future_truth_excluded=True,redundant_zoh_invariant=True))
        np.savez_compressed(OUT/f'predictions_fold{fold}.npz',**result)
    pd.DataFrame(support).to_csv(OUT/'evaluated_pair_support.csv',index=False)
    trainpairs=pairs[pairs.split=='train'];evaluated={u for f in plan['folds'] for u in f['eligible_all_frozen_pairs']}
    trainpairs[~trainpairs.pair_uid.isin(evaluated)].assign(reason='cross_fold_background_no_joint_held_support').to_csv(OUT/'unevaluable_background_pairs.csv',index=False)
    save(OUT/'inference_checks.json',dict(passed=True,models_checked=len(checks),checks=checks,test_read=False,original_inputs_sha256=sha(SWAP/'results/inputs.npz')))

def build():
    cfg=json.loads((HERE/'config.json').read_text());pairs=pd.read_csv(OUT/'pairs.csv');cases=pd.read_csv(OUT/'cases.csv');x=np.load(SWAP/'results/inputs.npz')
    truth=pd.read_csv(ABLATION/'results/factual_curves.csv.gz').query("variant=='V0' and model=='GRU-Control' and seed==17")
    template=pd.read_csv(ABLATION/'results/internal_curves.csv.gz').query("variant=='V0' and model=='GRU-Control' and seed==17")
    plan=json.loads((OUT/'holdout_protocol.json').read_text());facts=[];internal=[];windows=[]
    for f in plan['folds']:
        fold=f['fold'];pred=np.load(OUT/f'predictions_fold{fold}.npz');lookup={ci:i for i,ci in enumerate(pred['case_ids'])}
        for pi,pair in pairs[pairs.pair_uid.isin(set(f['eligible_all_frozen_pairs'])|{'validation:44'})].iterrows():
            real=truth[truth.pair_uid==pair.pair_uid];tt=real[['time_ms','time_kind','time_label']].drop_duplicates();at=tt.time_ms.to_numpy()/1000
            ridx={(r.time_label,r.variable,r.method):r for r in real.itertuples()};tidx={(r.time_label,r.variable):r for r in template[template.pair_uid==pair.pair_uid].itertuples()}
            for key in pred.files:
                if key=='case_ids':continue
                name,seed=key.split('_');ys=[];slow=[]
                for ci in range(pi*4,pi*4+4):
                    n=int(x['valid_steps'][ci]);time=x['native_time_us'][ci,:n+1]*1e-6;y=np.vstack([x['initial'][ci],pred[key][lookup[ci],:n]]).astype(float);y-=y[0].copy()
                    ys.append(old.tc.interp(time,y[:,CHANNELS],at));slow.append(old.tc.boxmean(time,y[:,CHANNELS],at,.27))
                    for j,r in enumerate(tt.itertuples()):
                        if r.time_kind!='grid' or round(r.time_ms) in cfg['report_times_ms']:
                            for k,var in enumerate(VARS):windows.append(dict(fold=fold,model=name,seed=int(seed),pair_uid=pair.pair_uid,case_id=ci,cell=cases.cell.iloc[ci],receiver_window=cases.receiver_window.iloc[ci],donor_window=cases.donor_window.iloc[ci],variable=var,time_label=r.time_label,time_ms=r.time_ms,increment=ys[-1][j,k]))
                aa,ab,ba,bb=ys;ca=aa-ab;cb=ba-bb;ha=aa-ba;hb=ab-bb;c=(ca+cb)/2;h=(ha+hb)/2;fact=aa-bb
                for ti,r in enumerate(tt.itertuples()):
                    for j,var in enumerate(VARS):
                        base=dict(fold=fold,model=name,seed=int(seed),pair_uid=pair.pair_uid,split=pair.split,role='heldout' if pair.split=='train' else 'selection_exposed_case',variable=var,time_ms=r.time_ms,time_kind=r.time_kind,time_label=r.time_label)
                        prev=tidx[(r.time_label,var)];row=dict(**base,supported=prev.supported,gA=prev.gA,gB=prev.gB,g=prev.g,real_F=prev.real_F)
                        for k,val in zip(['YAA','YAB','YBA','YBB','CA','CB','HA','HB','C','H','F','interaction'],[aa,ab,ba,bb,ca,cb,ha,hb,c,h,fact,ca-cb]):row[k]=val[ti,j] if prev.supported else np.nan
                        internal.append(row)
                        for method in ['A','B','D'] if j>=3 else ['A']:
                            prev=ridx[(r.time_label,var,method)];value=slow[0][ti,j]-slow[3][ti,j] if method=='D' else fact[ti,j]
                            assert np.isfinite(value) or not prev.supported
                            facts.append(dict(**base,method=method,supported=prev.supported,real=prev.real,pred=value if prev.supported else np.nan,error=value-prev.real if prev.supported else np.nan))
    facts=pd.DataFrame(facts);internal=pd.DataFrame(internal);facts.to_csv(OUT/'factual_curves.csv.gz',index=False);internal.to_csv(OUT/'internal_curves.csv.gz',index=False);pd.DataFrame(windows).to_csv(OUT/'window_points.csv.gz',index=False)
    selected=lambda d:d[(d.time_kind!='grid')|d.time_ms.round(6).isin(cfg['report_times_ms'])].copy()
    selected(facts).to_csv(OUT/'factual_points.csv.gz',index=False);selected(internal).to_csv(OUT/'internal_points.csv.gz',index=False)

def trajectory(device):
    raw=with_targets(load_numpy('train'),target_pack('train'));meta=pd.read_parquet(TRAIN/'train_windows.parquet');indices=np.load(OUT/'holdout_window_indices.npz');rows=[]
    with torch.inference_mode():
      for fold in range(3):
        ids=indices[f'fold{fold}_held_indices'];d=to_device(raw,device,ids)
        for name in COMBOS:
          m,_=load_model(fold,name,17,device)
          for ix in torch.arange(len(ids),device=device).split(2048):
            p=m(training.batch(d,ix));elapsed=d['dt'][ix].cumsum(1)
            for horizon in [.5,1.]:
              for reference,support in [('A','common_B_prefix'),('B','common_B_prefix'),('A','original_A_full')]:
                mask=(d['loss_mask'][ix] if support=='common_B_prefix' else d['future_mask'][ix])&(elapsed<=horizon+1e-7);count=mask.sum(1);assert (count>0).all();target=d['target_'+reference][ix];e=errors(p,target).square();axis=(p[...,CHANNELS]-target[...,CHANNELS]).square()
                mse=(e*mask[...,None]).sum(1)/count[:,None];am=(axis*mask[...,None]).sum(1)/count[:,None];last=count-1
                end=errors(p[torch.arange(len(ix),device=device),last],target[torch.arange(len(ix),device=device),last]);used=elapsed[torch.arange(len(ix),device=device),last]
                originals=ids[ix.cpu().numpy()];mm=meta.iloc[originals];mse_np=mse.cpu().numpy();axis_np=am.cpu().numpy();end_np=end.cpu().numpy();used_np=used.cpu().numpy()
                row=dict(fold=fold,model=name,seed=17,registered_index=originals,window_id=mm.window_id.to_numpy(),flight_id=mm.flight_id.to_numpy(),horizon_ms=round(horizon*1000),reference=reference,support=support,steps=count.cpu().numpy(),last_supported_s=used_np,endpoint_is_exact=abs(used_np-horizon)<1e-6)
                for k,var in enumerate(['position_m','velocity_mps','attitude_rad','rate_radps']):row[var+'_mse']=mse_np[:,k];row[var+'_endpoint_sq']=end_np[:,k]**2
                for k,var in enumerate(VARS):row[var+'_mse']=axis_np[:,k]
                rows.append(pd.DataFrame(row))

        del d;torch.cuda.empty_cache()
    out=pd.concat(rows,ignore_index=True);out.to_csv(OUT/'heldout_trajectory.csv.gz',index=False)
    keys=['fold','model','seed','flight_id','horizon_ms','reference','support'];metrics=[c for c in out if c.endswith('_mse') or c.endswith('_endpoint_sq')]
    flights=out.groupby(keys)[metrics+['last_supported_s']].mean();flights['windows']=out.groupby(keys).size();flights.to_csv(OUT/'trajectory_per_flight.csv')
    summ=[]
    for key,g in flights.reset_index().groupby(['model','horizon_ms','reference','support']):
      r=dict(zip(['model','horizon_ms','reference','support'],key),flights=g.flight_id.nunique(),windows=int(g.windows.sum()),mean_last_supported_s=g.last_supported_s.mean())
      for c in metrics:r[c.replace('_mse','_rms').replace('_endpoint_sq','_endpoint_rms')]=float(np.sqrt(g[c].mean()))
      summ.append(r)
    pd.DataFrame(summ).to_csv(OUT/'trajectory_summary.csv',index=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda:1');args=parser.parse_args();assert len(list((HERE/'folds').glob('fold*/training/*/seed*/summary.json')))==6
    device=setup(args.device,False);predict(device);build();trajectory(device)
