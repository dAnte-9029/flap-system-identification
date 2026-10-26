"""Training-fold marginal support only; no outcome-dependent exclusion or matching."""
from common import *

def write_validation_exclusions():
    pairs=pd.read_csv(OUT/'pairs.csv')
    pairs[(pairs.split=='validation')&(pairs.pair_uid!='validation:44')].assign(reason='non_primary_validation_background; not a training-flight holdout pair').to_csv(OUT/'excluded_validation_background.csv',index=False)

def main():
    raw=load_numpy('train');ids=np.load(OUT/'holdout_window_indices.npz');x=np.load(SWAP/'results/inputs.npz');cases=pd.read_csv(OUT/'cases.csv');plan=json.loads((OUT/'holdout_protocol.json').read_text());rows=[];refs=[]
    coordinates=['motor','common','differential','rudder']
    def transform(u):return np.stack([u[...,0],(u[...,1]+u[...,2])/2,(u[...,2]-u[...,1])/2,u[...,3]],-1)
    for f in plan['folds']:
        fi=f['fold'];fit=ids[f'fold{fi}_train_indices'];lo=np.full(4,np.inf);hi=-lo.copy();slo=np.full(6,np.inf);shi=-slo.copy();count=0
        for chunk in np.array_split(fit,100):
            events=raw['events'][chunk];length=raw['event_length'][chunk];valid=(np.arange(events.shape[2])[None,None,:]<length[...,None])&(events[...,4]>0)
            u=transform(events[...,:4][valid]);lo=np.minimum(lo,u.min(0));hi=np.maximum(hi,u.max(0));count+=len(u)
            state=raw['initial'][chunk][:,[3,4,5,10,11,12]];slo=np.minimum(slo,state.min(0));shi=np.maximum(shi,state.max(0))
        ref=dict(fold=fi,train_windows=len(fit),train_flights=len(f['train_flights']),valid_zoh_segments=count,control_coordinates=coordinates,control_min=lo.tolist(),control_max=hi.tolist(),initial_state_columns=['vx','vy','vz','p','q','r'],initial_min=slo.tolist(),initial_max=shi.tolist(),train_indices_sha256=hashlib.sha256(fit.tobytes()).hexdigest())
        refs.append(ref)
        for c in cases[cases.pair_uid.isin(set(f['eligible_all_frozen_pairs'])|{'validation:44'})].itertuples():
            i=c.case_id;ev=x['events'][i];valid=(np.arange(ev.shape[1])[None,:]<x['event_length'][i,:,None])&(ev[...,4]>0);u=transform(ev[...,:4][valid]);state=x['initial'][i,[3,4,5,10,11,12]];outside=(u<lo-1e-6)|(u>hi+1e-6)
            r=dict(fold=fi,case_id=i,pair_uid=c.pair_uid,cell=c.cell,receiver_window=c.receiver_window,donor_window=c.donor_window,control_outside_fold_marginal=bool(outside.any()),initial_outside_fold_marginal=bool(((state<slo-1e-6)|(state>shi+1e-6)).any()),outside_control_time_fraction=float(np.sum(ev[...,4][valid]*outside.any(1))/np.sum(ev[...,4][valid])),physical_range_violation=bool(((ev[...,:4][valid]<np.array([0,-1,-1,-1])-1e-6)|(ev[...,:4][valid]>1+1e-6)).any()),joint_support_confirmed=False)
            for k,name in enumerate(coordinates):r[name+'_min']=u[:,k].min();r[name+'_max']=u[:,k].max()
            rows.append(r)
            # All four cells preserve the receiver's complete non-control inputs.
            other=(i//4)*4+(0 if c.cell in ['AA','AB'] else 2)
            for key in ['history','history_mask','history_dt','history_length','initial','dt']:np.testing.assert_array_equal(x[key][i],x[key][other])
    pd.DataFrame(rows).to_csv(OUT/'fold_control_support.csv',index=False)
    save(OUT/'fold_support_reference.json',dict(reference='fit-fold training windows only; marginal coordinate extrema, not joint conditional support',folds=refs,original_swap_support='swap_support.csv is copied historical full-training audit, not a fold-local reference',excluded_by_support=0,test_read=False))
    save(OUT/'resolved_contract.json',dict(epochs=24,early_stopping=False,normalization='fold-local train indices only; source receipt and normalization_receipt.json are authoritative',unused_inherited_config_fields=['patience','input_output_normalization','evaluation_horizons_ms'],training_horizon_ms=1000,local_response_horizon_ms=500,M0_events='original GRU16',M1_M2_events='frozen two exact ZOH moments',config_sha256=sha(HERE/'config.json'),note='The listed fields are unused descriptive remnants from the historical configuration. No runtime settings or files changed during training.'))
    write_validation_exclusions()
    table=pd.DataFrame(rows).groupby('fold')[['control_outside_fold_marginal','initial_outside_fold_marginal','physical_range_violation']].agg(['sum','count']).to_string()
    print('\n'.join(line.rstrip() for line in table.splitlines()))

if __name__=='__main__':main()
