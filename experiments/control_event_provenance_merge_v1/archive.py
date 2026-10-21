import json,sys
from common import HERE,sha,save
OUT=HERE/'archive_manifest.json'
def paths():return [p for p in sorted(HERE.rglob('*')) if p.is_file() and p!=OUT and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]
if '--verify' in sys.argv:
 m=json.loads(OUT.read_text());assert {r['path'] for r in m['files']}=={str(p.relative_to(HERE)) for p in paths()}
 for r in m['files']:assert sha(HERE/r['path'])==r['sha256'],r['path']
 print('Archive verified',len(m['files']))
else:
 save(OUT,dict(base_commit='1ea494d1e497660b5a0c4053131613258a17862b',branch='exp/control-event-provenance-merge-v1',
  files=[dict(path=str(p.relative_to(HERE)),bytes=p.stat().st_size,sha256=sha(p)) for p in paths()]))
 print('Archived',len(paths()),'files')
