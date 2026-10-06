"""Staged diagnosis, input-only selection and single-factor training experiments."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from system_identification.models.reference_gru import ReferenceGRU
from evaluate_control_windows_v2 import error_rows, summarize

ROOT=base.ROOT
OLD=base.DATA
OUT=ROOT/'docs/analysis/results/control_windows_v3'
ART=ROOT/'artifacts/control_windows_v3'
SEEDS=[17,23,42]


def model_from(path):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    keys=['state_mean','state_scale','reference_mean','reference_scale','actuator_mean','actuator_scale','rate_scale','velocity_scale']
    model=ReferenceGRU({k:ck['state_dict'][k] for k in keys})
    model.load_state_dict(ck['state_dict']);model.eval()
    return model


def predict(model,x,indices,hold=False):
    predictions={k:[] for k in ['rates_b','velocity_n','quaternion']}
    with torch.no_grad():
        for start in range(0,len(indices),64):
            ix=indices[start:start+64]
            batch={k:v[ix] for k,v in x.items()}
            if hold:
                batch['future_reference']=batch['history_reference'][:,-1:, :].expand_as(batch['future_reference'])
            p=model(**batch)
            for k in predictions:
                assert torch.isfinite(p[k]).all()
                predictions[k].append(p[k].numpy())
    return {k:np.concatenate(v) for k,v in predictions.items()}


def diagnose():
    OUT.mkdir(exist_ok=False)
    base.write(OUT/'diagnostic_protocol.json',dict(seeds=SEEDS,source_manifest_sha256=base.sha(OLD/'manifest.json'),
        task='frozen A800 train/validation gap and observed-reference versus reference-held diagnostic; not causal intervention truth',
        horizons=[.1,.2,.5,1.],threshold_tuning_from_outputs=False,sealed_test_accessed=False))
    x,y,stats=base.load();windows=pd.read_csv(OLD/'windows_v2.csv')
    rows=[];sensitivity=[]
    for seed in SEEDS:
        model=model_from(ROOT/f'artifacts/control_windows_v2/A_{seed}/final.pt')
        for partition in ['train','validation']:
            m=windows[windows.partition==partition];ix=m.array_index.to_numpy()
            truth={k:v[ix].numpy() for k,v in y.items()}
            observed=predict(model,x,ix);held=predict(model,x,ix,hold=True)
            for name,pred in [('observed',observed),('reference_held',held)]:
                row=summarize(error_rows(pred,truth,m,name,seed));row['partition']=partition;rows.append(row)
            for step in [5,10,25,50]:
                for axis,name in enumerate(['p','q','r']):
                    delta=observed['rates_b'][:,step-1,axis]-held['rates_b'][:,step-1,axis]
                    err=observed['rates_b'][:,step-1,axis]-truth['rates_b'][:,step-1,axis]
                    sensitivity.append(dict(seed=seed,partition=partition,horizon_s=step*.02,axis=name,
                        predicted_reference_effect_rms=float(np.sqrt(np.mean(delta**2))),
                        prediction_error_rms=float(np.sqrt(np.mean(err**2)))))
    rows=pd.concat(rows,ignore_index=True);rows.to_csv(OUT/'diagnostic_per_flight.csv',index=False)
    rows.groupby(['partition','arm','subset','horizon_s'])[['p','q','r','velocity','attitude']].mean().to_csv(OUT/'diagnostic_summary.csv')
    pd.DataFrame(sensitivity).to_csv(OUT/'reference_sensitivity.csv',index=False)
    print(rows[(rows.subset=='all') & (rows.horizon_s==.5)].groupby(['partition','arm'])[['p','q','r','velocity','attitude']].mean())


def input_features(ref):
    """Only recorded inputs; no measured output, prediction error or response sign."""
    unit=np.array([np.deg2rad(1),np.deg2rad(1),.05,.03,.03])
    ranges=np.ptp(ref,axis=1)/unit
    net=np.abs(ref[:,-1]-ref[:,0])/unit
    path=np.abs(np.diff(ref,axis=1)).sum(1)/unit
    efficiency=net/np.maximum(path,1e-6)
    # Favor sizable sustained displacement over numerous quantization crossings.
    scores=np.max(ranges*(.5+.5*efficiency),axis=1)
    return scores,ranges,net,efficiency


def prepare():
    assert (OUT/'diagnostic_summary.csv').exists()
    assert not (OUT/'selection_protocol.json').exists()
    x,y,stats=base.load();w=pd.read_csv(OLD/'windows_v2.csv')
    score,ranges,net,eff=input_features(x['future_reference'].numpy())
    w['input_score']=score;w['max_scaled_range']=ranges.max(1)
    w['net_change_scaled']=net.max(1);w['max_input_efficiency']=eff.max(1)
    tr=w[w.partition=='train'];ids=[];thresholds=[]
    # Fixed top quartile separately within each training flight/mode, deterministic ties.
    for (log,mode),g in tr[tr.quality!='Q3'].groupby(['log_id','mode']):
        ranked=g.sort_values(['input_score','start_timestamp_us'],ascending=[False,True])
        n=max(1,int(np.ceil(len(ranked)*.25)));ids.extend(ranked.index[:n])
        thresholds.append(dict(log_id=log,mode=int(mode),kept=n,available=len(ranked),threshold=float(ranked.input_score.iloc[n-1])))
    c=w.loc[sorted(ids)];rng=np.random.default_rng(1309);b=[]
    for (log,mode),g in c.groupby(['log_id','mode']):
        pool=tr[(tr.log_id==log)&(tr['mode']==mode)]
        b.extend(rng.choice(pool.index,len(g),replace=False).tolist())
    for arm,indices in [('A',tr.index),('A_long',tr.index),('B',sorted(b)),('C',c.index),('D',tr.index)]:
        frame=w.loc[indices].copy()
        weight=np.where(frame.index.isin(c.index),3.,1.) if arm=='D' else np.ones(len(frame))
        for log in frame.log_id.unique():
            mask=frame.log_id.to_numpy()==log;weight[mask]/=weight[mask].sum()
        frame['sampling_probability']=weight/weight.sum();frame.to_csv(OUT/f'arm_{arm}.csv',index=False)
    w.to_csv(OUT/'windows_v3.csv',index=False)
    w[w.partition=='validation'].to_csv(OUT/'evaluation_windows.csv',index=False)
    protocol=dict(source_windows_sha256=base.sha(OLD/'windows_v2.csv'),
        source_manifest_sha256=base.sha(OLD/'manifest.json'),reference_model_input_contract='unchanged v2',
        score='max_axis(range/unit * (0.5+0.5*abs(net change)/total variation)); input-only',
        selection='top25percent per TRAIN flight/mode, Q3 excluded only from C; deterministic time tie-break',
        thresholds=thresholds,counts=dict(A=len(tr),B=len(b),C=len(c),D=len(tr)),
        BC_overlap=len(set(b)&set(c.index)),D='flight equal then high-input group3:others1; keep hard/quiet',
        fixed_evaluation='same739 validation windows, no recuration; no new blind test',
        sampling_units='retain original disjoint H26/K50 supports',train_validation_reassigned=False,
        normalization='reuse frozen A-train v2 stats, unchanged full A cohort',sealed_test_accessed=False,
        adaptation_notice='new hypothesis after weak-contrast v2 results; not an untouched confirmatory test')
    base.write(OUT/'selection_protocol.json',protocol)
    (OUT/'training').mkdir()
    training=dict(seeds=SEEDS,architecture='unchanged ReferenceGRU64, 40070 parameters',
        data_strategy_experiment='reuse frozen A800; B/C/D800 each seed, same v2 hyperparameters',
        convergence_experiment='A_long1600 from same initialization; only update budget doubles',
        checkpoint='fixed final; all3 seeds evaluated; no best-seed/epoch choice',
        batch=32,lr=.0003,weight_decay=.0001,gradient_clip=1.,
        loss='identical v2 rate + velocity + 0.2 attitude loss',
        primary_candidate='A_long predeclared for complete-context comparison to A; not best validation arm',
        relative_gate='same5percent improvement and seed/flight uncertainty gates for data strategies; absolute fitness separate',
        absolute_research_gate='at0.5s roll-active p and pitch-active q each flight-equal RMSE<=0.15rad/s; all attitude<=3deg; velocity component RMSE<=0.5m/s; all3 seeds',
        tolerances_status='engineering research targets set before new training, not platform-validated safety bounds',
        response_gate='full-context rollout and rolling1s diagnostics separate; >=10complete clusters each sign and >=3flights per target axis; waveform amplitude error improves>=10percent over A; timing median<=max40ms,2*native interval; no direct causal-effect claim',
        control_gate='no optimization or deployment unless absolute and response evidence gates pass; physics/hardware safety not inferred',
        script_sha256=base.sha(__file__),model_sha256=base.sha(ROOT/'src/system_identification/models/reference_gru.py'),
        parent_protocol_sha256=base.sha(OLD/'training/protocol.json'),selection_sha256=base.sha(OUT/'selection_protocol.json'))
    base.write(OUT/'training/protocol.json',training)
    print(json.dumps(protocol,indent=2))


def train():
    assert (OUT/'training/protocol.json').exists()
    assert not (OUT/'training/status.json').exists()
    x,y,stats=base.load()
    base.DATA=OUT;base.OUT=OUT/'training';base.ART=ART
    # Reused helper's checkpoint provenance points to this stage's manifest.
    base.write(OUT/'manifest.json',dict(parent_manifest_sha256=base.sha(OLD/'manifest.json'),
        training_protocol_sha256=base.sha(OUT/'training/protocol.json'),selection_sha256=base.sha(OUT/'selection_protocol.json'),
        table_hashes={p.name:base.sha(p) for p in OUT.glob('*.csv')},sealed_test_accessed=False))
    completed=[]
    try:
        for arm,steps in [('B',800),('C',800),('D',800),('A_long',1600)]:
            for seed in SEEDS:
                base.write(OUT/'training/status.json',dict(status='running',arm=arm,seed=seed,completed=completed))
                completed.append(base.train_one(arm,seed,steps,x,y,stats))
        base.write(OUT/'training/status.json',dict(status='complete',completed=completed))
    except Exception as exc:
        base.write(OUT/'training/status.json',dict(status='failed',error=repr(exc),completed=completed));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['diagnose','prepare','train']);args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    {'diagnose':diagnose,'prepare':prepare,'train':train}[args.stage]()
