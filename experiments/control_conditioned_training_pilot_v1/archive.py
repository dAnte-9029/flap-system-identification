from common import *
import platform,subprocess,torch,scipy
from datetime import datetime,timezone

def main():
 h=pd.read_csv(OUT/'training_history.csv');summary=pd.read_csv(OUT/'training_summary.csv');assert len(h)==72 and len(summary)==12 and summary.updates.sum()==6384
 batches=[]
 for p in sorted((HERE/'folds').glob('fold*/training/*/seed17/training.jsonl')):
  for line in p.read_text().splitlines():
   r=json.loads(line)
   for b in r['batch_weight_statistics']:batches.append(dict(fold=r['fold'],model=r['model'],epoch=r['epoch'],**b))
 pd.DataFrame(batches).to_csv(OUT/'batch_weight_statistics.csv.gz',index=False)
 save(OUT/'environment.json',dict(python=sys.executable,platform=platform.platform(),torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,cuda=torch.version.cuda,recorded_utc=datetime.now(timezone.utc).isoformat(),dependencies_modified=False,training_device='cuda:1',training_tf32=True,evaluation_tf32=False,gpu_after=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv'],text=True),test_read=False))
 files=[]
 for p in sorted(HERE.rglob('*')):
  if p.is_file() and p.name!='archive_manifest.json' and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts:files.append(dict(path=str(p.relative_to(HERE)),bytes=p.stat().st_size,sha256=sha(p)))
 save(HERE/'archive_manifest.json',dict(base_sha='73c03c73237aa0d6e6c7608827263a94f696f793',branch='exp/control-conditioned-training-pilot-v1',files=files,training_runs=12,epochs=72,updates=6384,test_read=False))
 print('Archived',len(files),'files; 12 runs, 72 epochs, 6384 updates')
if __name__=='__main__':main()
