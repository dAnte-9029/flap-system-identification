"""Hash only this branch's changed/new code and experiment artifacts."""
import json
import subprocess
import sys
from run import ROOT, HERE, sha, save

OUT=HERE/'archive_manifest.json'


def paths():
    changed=subprocess.check_output(['git','diff','--name-only','653b60d'],cwd=ROOT,text=True).splitlines()
    new=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
    return [ROOT/name for name in sorted(set(changed+new))
            if ROOT/name!=OUT and (ROOT/name).is_file()
            and '__pycache__' not in name and '.pytest_cache' not in name]


if '--verify' in sys.argv:
    m=json.loads(OUT.read_text())
    assert {str(p.relative_to(ROOT)) for p in paths()}=={r['path'] for r in m['files']}
    for row in m['files']:assert sha(ROOT/row['path'])==row['sha256'],row['path']
    print('Archive verified:',len(m['files']),'files')
else:
    save(OUT,dict(base_commit='653b60d',branch='fix/control-event-canonicalization-v1',
        files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in paths()]))
    print('Archived:',len(paths()),'files')
