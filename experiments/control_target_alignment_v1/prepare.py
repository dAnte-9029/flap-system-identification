from common import *
import platform
cfg=json.loads((HERE/'config.json').read_text());assert sha(DATA/'dataset_manifest.json')==cfg['manifest_sha256']
m=json.loads((DATA/'dataset_manifest.json').read_text());sources={}
def add(p,expected=None):
 h=sha(p)
 if expected:assert h==expected,(p,h,expected)
 sources[str(p)]=h
add(DATA/'dataset_manifest.json',cfg['manifest_sha256'])
for split in cfg['splits']:
 for stem in ['samples','native_vehicle_angular_velocity']:
  name=f'{stem}_{split}.parquet';add(DATA/name,m['artifact_sha256'][name])
 receipt=json.loads((PILOT/'results/source_receipt.json').read_text())['files']
 for p in [CACHE/f'{split}.npz',TRAIN/f'{split}_windows.parquet']:
  expected=next(v for k,v in receipt.items() if k.endswith('/'+p.name) and (str(CACHE) in k if p.suffix=='.npz' else 'horizon_training_v1' in k));add(p,expected)
for exp,files in [(OLD,['native_rollouts.npz','windows.csv','pairs.csv','cohorts.csv']),(TIMING,['native_diagnostic_inputs.npz','native_rate_extract.csv.gz','origin_clock_audit.csv','future_clock_audit.csv']),(PILOT,['factual_points.csv.gz','factual_curves.csv.gz','factual_summary.csv','component_assignments.csv','predictions_fold0.npz','predictions_fold1.npz','predictions_fold2.npz']),(FORMAL,['factual_points.csv.gz','factual_summary.csv','predictions_fold0.npz','predictions_fold1.npz','predictions_fold2.npz']),(SWAP,['inputs.npz','cases.csv','pairs.csv'])]:
 archive=json.loads((exp/'archive_manifest.json').read_text());entries=archive.get('files',[])
 if isinstance(entries,list):known={r['path']:r['sha256'] for r in entries}
 else:known={k:(v['sha256'] if isinstance(v,dict) else v) for k,v in entries.items()}
 for name in files:add(exp/'results'/name,known.get('results/'+name))
for name in ['experiments/local_control_response_timing_slowfast_v1/common.py','experiments/local_control_response_timing_slowfast_v1/analyze.py','experiments/local_control_response_rollout_v1/analyze.py','scripts/prepare_velocity_rate_learning.py','scripts/train_velocity_rate_learning.py','src/system_identification/data/velocity_rate_dataset.py','src/system_identification/models/velocity_rate_gru.py','experiments/local_control_response_time_contract_ablation_v1/time_contract_zh.md','configs/data/velocity_rate_dataset_registry.yaml']:add(ROOT/name)
z,n,w,p,c=read_assets();assert len(w)==225 and len(p)==113
assert set(w.split)=={'train','validation'}
for r in p.itertuples():
 for i,key,f in [(r.a,r.window_a,r.flight_a),(r.b,r.window_b,r.flight_b)]:
  q=w.iloc[i];assert q.window_id==key and q.flight_id==f and q.split==r.split
assign=pd.read_csv(PILOT/'results/component_assignments.csv');assert len(assign)==48 and assign.component_id.nunique()==11
assert set(assign.pair_uid)==set(c.query("split=='train' and cohort=='main'").pair_uid)
for name,d in [('windows',w),('pairs',p),('cohorts',c),('component_assignments',assign)]:d.to_csv(OUT/f'{name}.csv',index=False)
save(OUT/'source_receipt.json',dict(base_sha=cfg['base_sha'],dataset_manifest=cfg['manifest_sha256'],files=sources,partitions=['train','validation'],test_read=False,training_runs=0,checkpoint_loaded=False,python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,fixed_windows=len(w),fixed_pairs=len(p),main_pairs=48,components=11))
print('Source checks passed',len(sources),flush=True)
