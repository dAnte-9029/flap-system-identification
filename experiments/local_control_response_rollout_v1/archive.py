"""Hash or verify this experiment only; never traverses dataset directories."""
from pathlib import Path
import argparse
import hashlib
import json

HERE=Path(__file__).resolve().parent

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def files():
    return sorted(p for p in HERE.rglob('*') if p.is_file() and p.name!='archive_manifest.json'
                  and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--verify',action='store_true');args=ap.parse_args()
    path=HERE/'archive_manifest.json'
    if args.verify:
        manifest=json.loads(path.read_text())
        assert {x['path'] for x in manifest['files']}=={str(p.relative_to(HERE)) for p in files()}
        for row in manifest['files']:
            p=HERE/row['path'];assert p.stat().st_size==row['bytes'] and digest(p)==row['sha256'],p
        print('archive verified:',len(manifest['files']),'files')
    else:
        result={'base_sha':'4fecb63ad65b1314227d168ea5457676b46752d7','branch':'exp/local-control-response-rollout-v1',
                'scope':'Only this experiment; original assets referenced by input receipts',
                'files':[dict(path=str(p.relative_to(HERE)),bytes=p.stat().st_size,sha256=digest(p)) for p in files()]}
        path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print('archive written:',len(result['files']),'files')


if __name__=='__main__':main()
