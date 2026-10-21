"""Content-address this local diagnosis without changing its sources."""
import json,hashlib,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=HERE/'archive_manifest.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def paths():return [p for p in sorted(HERE.rglob('*')) if p.is_file() and p!=OUT and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]
if '--verify' in sys.argv:
    m=json.loads(OUT.read_text());assert {r['path'] for r in m['artifacts']}=={str(p.relative_to(HERE)) for p in paths()}
    for r in m['artifacts']:assert sha(HERE/r['path'])==r['sha256'] and (HERE/r['path']).stat().st_size==r['bytes'],r['path']
    print('Archive verified:',len(m['artifacts']),'files')
else:
    OUT.write_text(json.dumps(dict(base_commit='8455e9f785f559ea455e18d1760058bfc0ff20f9',branch='exp/control-event-invariance-check',
        code_version='Git commit containing this archive; individual source hashes below',
        artifacts=[dict(path=str(p.relative_to(HERE)),sha256=sha(p),bytes=p.stat().st_size) for p in paths()]),indent=2)+'\n')
    print('Archived',len(paths()),'files')
