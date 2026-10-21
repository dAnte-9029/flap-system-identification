from pathlib import Path
import sys,json,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results'
TRAIN=ROOT/'experiments/velocity_rate_gru64_horizon_training_v1'
CACHE=Path('/home/zn/flap-system-identification/.worktrees/velocity-rate-dataset-v1/artifacts/velocity_rate_gru64_horizon_training_v1')
DATA=Path('/home/zn/flap-system-identification/dataset/velocity_rate_multistep_v1')
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=lambda v:v.item() if hasattr(v,'item') else v.isoformat())+'\n')
def cache_hash(split):
 assert split in ['train','validation'];expected=json.loads((TRAIN/'data_audit.json').read_text())['partitions'][split]['cache_sha256']
 assert sha(CACHE/f'{split}.npz')==expected;return expected
