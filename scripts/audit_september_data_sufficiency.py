"""Inventory + fixed command-support audit, no models fitted or evaluated."""
from pathlib import Path
import json,re,subprocess
import numpy as np
import pandas as pd
import torch,yaml
import run_response_timescale_diagnostic as d
m=d.m;ROOT=d.ROOT
OUT=ROOT/'docs/analysis/results/september_data_sufficiency_v1'

def nonoverlap(rows):
    chosen=[];end=-np.inf
    for _,r in rows.sort_values('origin_timestamp_us',kind='stable').iterrows():
        if r.origin_timestamp_us>=end:
            chosen.append(r.window_id);end=r.end_us
    return chosen

def run():
    assert not OUT.exists();OUT.mkdir(parents=True)
    inventory=ROOT/'docs/analysis/results/raw_log_inventory_20260920/logs.csv'
    manifest=ROOT/'dataset/trajectory_v3_september_expanded/manifest.json'
    registry=yaml.safe_load((ROOT/'configs/data/trajectory_dataset_registry.yaml').read_text());record=registry['datasets'][registry['default_dataset_id']]
    assert record['manifest_path']==m.rel(manifest) and m.file_hash(manifest)==record['manifest_sha256']
    old=m.read_json(manifest);inv=pd.read_csv(inventory)
    covpath=ROOT/'docs/analysis/results/paper_independent_test_v1/test_flight_coverage.csv';cov=pd.read_csv(covpath)
    prep=ROOT/'artifacts/paper_independent_test_v1/prepared.pt';statuspath=ROOT/'docs/analysis/results/paper_independent_test_v1/preparation_status.json'
    status=m.read_json(statuspath);assert m.file_hash(prep)==status['prepared_sha256']
    threshold=ROOT/'docs/analysis/results/response_timescale_diagnostic_v1/channel_thresholds.csv';th=pd.read_csv(threshold)
    inputs=[inventory,manifest,covpath,statuspath,threshold,prep,m.ART/'prepared.pt',ROOT/'docs/analysis/results/conditional_excitation_v1/report.md']
    pins={m.rel(p):m.file_hash(p) for p in inputs}
    m.write_json(OUT/'protocol.json',dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      scope='September6-19 existing archive inventory; unchanged admitted development and previously opened test command windows; no model predictions or performance files read',
      source_dataset_id=old['dataset_id'],pins=pins,
      test_access='Current user explicitly requests Sep6-Sep19 data sufficiency. Previously opened Sep8/Sep19 prepared caches are accessed for command coverage. Roles unchanged; not added to training; not called untouched again.',
      selection='Reuse prior 500ms channel coordinate/scales/train q25/q75. high>q75; isolated=high & all other channels<=q25; persistence=abs(mean departure)/RMS>=0.8.',
      counting='Raw window counts plus greedy earliest nonoverlapping full50-step intervals within flight and segment, separately per channel. Nonoverlap is NOT statistical independence or randomized exogeneity.',
      interpretation='Sufficiency distinguished for logged-distribution forecasting vs sustained/counterfactual command responses. Isolated rules not universal identifiability test. No binary sample-count success threshold.'))
    # Hash-level inventory mapping handles aliases and duplicated files.
    roles={}
    for split in ['train','validation']:
        for flight in old['split_contract']['assignments'][split]:roles[old['source']['ulog_sha256'][flight]]=(split,flight,'admitted')
    for _,row in cov.iterrows():roles[row.sha256]=(row.date,row.flight_id,row.status)
    inv['role']=inv.sha256.map(lambda h:roles.get(h,('not_admitted','', 'existing_exclusion_or_unassigned'))[0])
    inv['flight_id']=inv.sha256.map(lambda h:roles.get(h,('','',''))[1])
    inv['current_status']=inv.sha256.map(lambda h:roles.get(h,('','','existing_exclusion_or_unassigned'))[2])
    inv.to_csv(OUT/'inventory.csv',index=False)
    unique=inv.drop_duplicates('sha256');date=unique.groupby(['filename_date','role','current_status']).agg(unique_logs=('sha256','size')).reset_index();date.to_csv(OUT/'date_inventory.csv',index=False)
    sp,batches,stats=d.r.source.inputs()
    with (OUT/'access_log.jsonl').open('a') as f:f.write(json.dumps(dict(time_utc=pd.Timestamp.now(tz='UTC').isoformat(),path=m.rel(prep),content='previously opened Sep8/Sep19 cached controls and identities, no predictions'))+'\n')
    held=torch.load(prep,map_location='cpu',weights_only=False);batches={**batches,**held}
    rows=[];selected=[];coverage=[]
    for split,b in batches.items():
        assert split in ['train','validation','Sep8','Sep19']
        t=b.trajectory
        origins=pd.read_csv(m.BASE/f'{split}_origins.csv' if split in ['train','validation'] else ROOT/f'artifacts/paper_independent_test_v1/origins_{split}.csv')
        np.testing.assert_array_equal(origins.window_id,t.window_ids)
        score,mean,pers=d.channel_scores(t.controls,th.scale.to_numpy());masks=d.isolated_masks(score,th.train_q25.to_numpy(),th.train_q75.to_numpy())
        origins['end_us']=origins.origin_timestamp_us+np.sum(t.dt_s,axis=1)*1e6
        for chindex,ch in enumerate(d.r.CHANNELS):
            isolated=masks[ch]['isolated'];sustained=isolated&(pers[:,chindex]>=.8)
            for subset,mask in [('high',masks[ch]['high']),('isolated',isolated),('isolated_one_direction',sustained)]:
                f=origins[mask].copy();f['sign']=np.sign(mean[mask,chindex]);f['persistence']=pers[mask,chindex]
                for flight,g in f.groupby('log_id'):
                    chosen=[]
                    for _,segment in g.groupby('segment_id'):chosen+=nonoverlap(segment)
                    pick=g[g.window_id.isin(chosen)]
                    rows.append(dict(partition=split,channel=ch,subset=subset,flight_id=flight,n_windows=len(g),positive=int((g.sign>0).sum()),negative=int((g.sign<0).sum()),
                        n_nonoverlap=len(pick),nonoverlap_positive=int((pick.sign>0).sum()),nonoverlap_negative=int((pick.sign<0).sum())))
                    if subset=='isolated_one_direction':
                        for _,p in pick.iterrows():selected.append(dict(partition=split,channel=ch,flight_id=flight,window_id=p.window_id,origin_timestamp_us=p.origin_timestamp_us,end_us=p.end_us,sign=p.sign,persistence=p.persistence))
                k=pd.DataFrame([x for x in rows if x['partition']==split and x['channel']==ch and x['subset']==subset])
                coverage.append(dict(partition=split,channel=ch,subset=subset,total_origins=len(t.window_ids),admitted_flights=len(set(t.log_ids)),
                    flights=len(k),**{c:int(k[c].sum()) if len(k) else 0 for c in ['n_windows','positive','negative','n_nonoverlap','nonoverlap_positive','nonoverlap_negative']},
                    flights_both_signs=int(((k.nonoverlap_positive>0)&(k.nonoverlap_negative>0)).sum()) if len(k) else 0))
    pd.DataFrame(rows).to_csv(OUT/'excitation_per_flight.csv',index=False);pd.DataFrame(coverage).to_csv(OUT/'excitation_coverage.csv',index=False);pd.DataFrame(selected).to_csv(OUT/'nonoverlap_candidates.csv',index=False)
    previous=pd.read_csv(ROOT/'docs/analysis/results/response_timescale_diagnostic_v1/excitation_coverage.csv').query('cohort=="ALL" and subset=="isolated"')
    current=pd.DataFrame(coverage).query('partition in ["train","validation"] and subset=="isolated"')
    joined=previous.merge(current,on=['partition','channel','subset']);assert len(joined)==8;assert (joined.n_origins==joined.n_windows).all()
    m.verify_pins(pins)
    m.write_json(OUT/'checks.json',dict(previous_train_val_isolated_counts_exact=True,inventory_files=len(inv),unique_hashes=len(unique),admitted_flights=86,
      heldout_cache_accessed_this_run=True,heldout_predictions_or_metrics_accessed_this_run=False,no_training=True,no_split_changes=True,source_hashes_unchanged=True))

if __name__=='__main__':run()
