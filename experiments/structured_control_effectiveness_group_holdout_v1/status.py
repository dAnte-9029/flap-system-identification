from common import *
def main():
    summaries=list((HERE/'folds').glob('fold*/training/*/seed*/summary.json'))
    print('Completed',len(summaries),'/ 27')
    for folder in sorted((HERE/'jobs').glob('job*')):
        owner=json.loads((folder/'owner.json').read_text()) if (folder/'owner.json').exists() else {}
        if (folder/'done.json').exists():print(folder.name,owner.get('fold'),owner.get('seed'),'done');continue
        print(folder.name,owner)
        for name in ['M0','M1','M2']:
            p=HERE/'folds'/f"fold{owner.get('fold')}"/'training'/name/f"seed{owner.get('seed')}"/'training.jsonl'
            if p.exists():
                r=json.loads(p.read_text().splitlines()[-1]);print(name,'epoch',r['epoch'],'seconds',round(r['seconds'],2),'val',round(r['validation_flight_loss'],5))
        if (folder/'failure.json').exists():print('FAILURE',(folder/'failure.json').read_text())
if __name__=='__main__':main()
