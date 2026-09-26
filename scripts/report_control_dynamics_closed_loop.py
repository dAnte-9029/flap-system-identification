"""Failure-inclusive feedback/replay summary; no new controller or model choice."""
from pathlib import Path
import numpy as np
import pandas as pd
import run_paper_rollout_horizon_ablation as source

m=source.m
ROOT=m.ROOT/'docs/analysis/results/control_dynamics_realdata_v1'
IN=ROOT/'px4_guarded_closed_loop'
OUT=ROOT/'px4_guarded_summary'


def run():
    completion=m.read_json(IN/'completion.json')
    m.verify_pins(completion['outputs'])
    OUT.mkdir(parents=True,exist_ok=False)
    f=pd.read_csv(IN/'per_origin.csv')
    keys=['model','source_model','mode','offset','steps']
    rows=[]
    for key,g in f.groupby(keys):
        complete=g[g.completed]
        rows.append(dict(zip(keys,key),requested=len(g),admitted=int(g.admitted.sum()),completed=int(g.completed.sum()),
            completion_fraction=float(g.completed.mean()),accepted_steps_mean=float(g.accepted_steps.mean()),
            completed_only_flight_rmse=float(complete.groupby('log_id').rmse.mean().mean()) if len(complete) else None,
            completed_flights=complete.log_id.nunique(),guard_fraction=float(g.guard_fraction.mean())))
    pd.DataFrame(rows).to_csv(OUT/'survival.csv',index=False)
    idcols=['model','offset','origin','window_id','log_id','steps']
    hold=f[f['mode']=='open_hold'][idcols+['completed','rmse']]
    fb=f[f['mode']=='px4'][idcols+['completed','rmse']]
    own=fb.merge(hold,on=idcols,suffixes=('_feedback','_hold'),validate='one_to_one')
    own['paired_complete']=own.completed_feedback&own.completed_hold
    own['effect']=np.where(own.paired_complete,own.rmse_feedback-own.rmse_hold,np.nan)
    own['beneficial_and_complete']=own.paired_complete&(own.effect<0)
    own.to_csv(OUT/'own_feedback_per_origin.csv',index=False)
    own.groupby(['model','offset','steps']).agg(requested=('origin','size'),paired_complete=('paired_complete','sum'),
        beneficial_and_complete=('beneficial_and_complete','sum')).to_csv(OUT/'own_feedback_counts.csv')
    # Four trajectories must complete before effect-sign transfer is assessable.
    rep=f[f['mode']=='replay'][['model','source_model','offset','origin','window_id','log_id','steps','completed','rmse']]
    dest=rep.merge(hold,on=idcols,suffixes=('_replay','_hold'),validate='many_to_one')
    src=own.rename(columns={'model':'source_model','paired_complete':'source_complete','effect':'source_effect'})
    dest=dest.merge(src[['source_model','offset','origin','steps','source_complete','source_effect']],on=['source_model','offset','origin','steps'],validate='many_to_one')
    dest['four_complete']=dest.completed_replay&dest.completed_hold&dest.source_complete
    dest['destination_effect']=np.where(dest.four_complete,dest.rmse_replay-dest.rmse_hold,np.nan)
    dest['benefit_reversed']=dest.four_complete&(dest.source_effect<0)&(dest.destination_effect>0)
    dest.to_csv(OUT/'transfer_per_origin.csv',index=False)
    dest.groupby(['source_model','model','offset','steps']).agg(requested=('origin','size'),four_complete=('four_complete','sum'),
        source_benefit_destination_harm=('benefit_reversed','sum')).to_csv(OUT/'transfer_counts.csv')
    # Common-duration metric includes partially surviving pairs; duration is explicit.
    traces=pd.read_csv(IN/'traces.csv')
    grouped={key:g.sort_values('step') for key,g in traces.groupby(['model','source_model','mode','offset','origin'])}
    common=[]
    for (model,src,mode,offset,origin),a in grouped.items():
        if mode!='px4':continue
        b=grouped[(model,model,'open_hold',offset,origin)]
        valid=a.accepted_step.to_numpy()&b.accepted_step.to_numpy()
        for k in (5,10,25,50):
            mask=valid[:k];n=int(mask.sum())
            common.append(dict(model=model,offset=offset,origin=origin,steps=k,common_steps=n,
                feedback_rmse=float(np.sqrt(np.mean(a.error.to_numpy()[:k][mask]**2))) if n else None,
                hold_rmse=float(np.sqrt(np.mean(b.error.to_numpy()[:k][mask]**2))) if n else None))
    pd.DataFrame(common).to_csv(OUT/'paired_common_duration.csv',index=False)
    m.write_json(OUT/'completion.json',dict(status='complete',heldout_accessed=False,
        source_completion_sha256=m.file_hash(IN/'completion.json'),
        method='survival/all requested first; complete-pair and common-duration effects separate; replay effect requires source feedback/hold and destination replay/hold all complete',
        script_sha256=m.file_hash(Path(__file__)),outputs={m.rel(p):m.file_hash(p) for p in OUT.iterdir() if p.is_file()}))


if __name__=='__main__':run()
