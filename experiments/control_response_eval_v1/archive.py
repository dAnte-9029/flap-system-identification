"""Archive or verify this experiment only; excludes its own manifest and interpreter caches."""
import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
MANIFEST=HERE/'results/archive_manifest.json'
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def artifacts():
    return [p for p in sorted(HERE.rglob('*')) if p.is_file() and p!=MANIFEST and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts]
if '--verify' in sys.argv:
    m=json.loads(MANIFEST.read_text())
    assert {x['path'] for x in m['artifacts']}=={str(p.relative_to(HERE)) for p in artifacts()}
    for r in m['artifacts']:
        p=HERE/r['path'];assert p.stat().st_size==r['bytes'] and digest(p)==r['sha256'],str(p)
    print('Archive verification passed:',len(m['artifacts']),'files')
else:
    records=[dict(path=str(p.relative_to(HERE)),bytes=p.stat().st_size,sha256=digest(p)) for p in artifacts()]
    MANIFEST.write_text(json.dumps(dict(base_commit='1745345d06749c6b127e43660a9bf1947d058bea',
        branch='exp/control-response-eval-v1',code_version='Git commit containing this manifest; individual code hashes below',
        test_data_loaded=False,retrained=False,artifacts=records),indent=2)+'\n')
    print('Archived',len(records),'files;',sum(x['bytes'] for x in records),'bytes')
