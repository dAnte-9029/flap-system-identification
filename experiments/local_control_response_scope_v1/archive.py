"""Create a nonrecursive manifest for the new experiment only."""
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import numpy
import pandas
import scipy
import pyarrow

HERE=Path(__file__).resolve().parent


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    records=[]
    for p in sorted(HERE.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.name=='archive_manifest.json':continue
        records.append(dict(path=str(p.relative_to(HERE)),bytes=p.stat().st_size,sha256=sha(p),
                      storage='local_reproducible' if p.name=='all_eligible_metadata.csv.gz' or p.name.endswith('_report.txt') else 'git'))
    result=dict(base_sha='91f772448b78fd8690df83b9f98177d1b897218e',
                branch=subprocess.check_output(['git','branch','--show-current'],cwd=HERE,text=True).strip(),
                publication='See containing Git commit and remote verification; manifest cannot contain its own commit SHA',
                environment=dict(python=platform.python_version(),numpy=numpy.__version__,pandas=pandas.__version__,
                                 scipy=scipy.__version__,pyarrow=pyarrow.__version__),
                test_opened=False,model_loaded=False,files=records)
    (HERE/'archive_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(files=len(records),git_bytes=sum(x['bytes'] for x in records if x['storage']=='git'))))


if __name__=='__main__':main()
