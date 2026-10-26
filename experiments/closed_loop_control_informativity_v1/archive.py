from extract import *
import subprocess

def main():
 files=[]
 for p in sorted(HERE.rglob('*')):
  if not p.is_file() or '__pycache__' in p.parts or '.pytest_cache' in p.parts or p.name=='archive_manifest.json':continue
  rel=str(p.relative_to(HERE));local=rel in ['results/causal_features.parquet','results/diagnostic_arrays.npz'] or p.name.endswith('.local.txt') or p.name in ['Dinkla_published.pdf','Dinkla_published.txt']
  files.append(dict(path=rel,bytes=p.stat().st_size,sha256=old.sha(p),publication='local_rebuild_or_copyright_cache' if local else 'committed'))
 write_json(HERE/'archive_manifest.json',dict(base_sha=CFG['base_sha'],files=files,test_read=False,new_dynamics_training=False))
 print('files',len(files),'committed MB',sum(x['bytes'] for x in files if x['publication']=='committed')/1e6)
if __name__=='__main__':main()
