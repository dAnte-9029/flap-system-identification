"""Hash this audit's code, report and results without touching any dataset artifacts."""
import subprocess
from audit import HERE,ROOT,sha,write_json

def main():
    records=[]
    for f in sorted(HERE.rglob('*')):
        if not f.is_file() or '__pycache__' in f.parts or '.pytest_cache' in f.parts:continue
        rel=f.relative_to(HERE).as_posix()
        if rel=='results/archive_manifest.json':continue
        local_only=f.suffix=='.parquet' and f.name.startswith(('pairs_','windows_'))
        records.append(dict(path=rel,bytes=f.stat().st_size,sha256=sha(f),storage='local_reproducible_archive' if local_only else 'git'))
    write_json(HERE/'results/archive_manifest.json',dict(
        base_commit='8964b44933c7364c066d3cb09440a6d2a7f558f1',
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        artifacts=records,total_bytes=sum(r['bytes'] for r in records),
        local_only_bytes=sum(r['bytes'] for r in records if r['storage'].startswith('local')),
        note='No self hash. Content hashes identify the executed code; final Git commit is the delivery commit.'))
    print('Archived',len(records),'files;',sum(r['bytes'] for r in records),'bytes')

if __name__=='__main__':main()
