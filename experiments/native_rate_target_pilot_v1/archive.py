from common import *
import platform,subprocess,torch
from datetime import datetime,timezone

def main():
 summaries=pd.read_csv(OUT/'training_summary.csv');assert len(summaries)==6 and summaries.updates.sum()==3192
 histories=[]
 for p in sorted((HERE/'folds').glob('fold*/training/*/seed17/training.jsonl')):histories.extend(json.loads(x) for x in p.read_text().splitlines())
 assert len(histories)==36;pd.DataFrame(histories).to_csv(OUT/'training_history.csv',index=False)
 save(OUT/'environment.json',dict(python=sys.executable,python_version=platform.python_version(),platform=platform.platform(),torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,cuda=torch.version.cuda,recorded_utc=datetime.now(timezone.utc).isoformat(),dependencies_modified=False,training_device='cuda:1',training_tf32=True,evaluation_tf32=False,gpu_after=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv'],text=True),test_read=False))
 files=[]
 for p in sorted(HERE.rglob('*')):
  if p.is_file() and p.name not in ('archive_manifest.json','archive.log') and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts:files.append(dict(path=str(p.relative_to(HERE)),bytes=p.stat().st_size,sha256=sha(p)))
 save(HERE/'archive_manifest.json',dict(base_sha='ac5301011dd288e29643bf85f199ce86f146b58d',branch='exp/native-rate-target-pilot-v1',files=files,training_runs=6,epochs=36,updates=3192,test_read=False))
 print('Archived',len(files),'files; 6 runs / 36 epochs / 3192 updates')
if __name__=='__main__':main()
