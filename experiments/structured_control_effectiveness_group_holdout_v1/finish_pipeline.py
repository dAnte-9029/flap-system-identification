"""Wait for all fixed-budget jobs, then execute evaluation and independent checks."""
from common import *
import time,subprocess,datetime

def main():
    assert sys.executable=='/home/zn/anaconda3/envs/flap-train-gpu/bin/python'
    while len(list((HERE/'jobs').glob('job*/done.json')))<9:
        failures=list((HERE/'jobs').glob('job*/failure.json'))
        if failures:raise RuntimeError(f'Training failure requires investigation: {failures}')
        print(datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed',len(list((HERE/'folds').glob('fold*/training/*/seed*/summary.json'))),'/27',flush=True)
        time.sleep(30)
    assert len(list((HERE/'folds').glob('fold*/training/*/seed*/summary.json')))==27
    commands=[('evaluation.log',[str(HERE/'evaluate.py'),'--device','cuda:1']),('analysis.log',[str(HERE/'analyze.py')]),('verification.log',[str(HERE/'verify.py')]),('plot.log',[str(HERE/'plot.py')]),('tests_final.log',['-m','pytest','-q',str(HERE/'test_contract.py'),str(HERE/'test_results.py')])]
    records=[]
    for log,args in commands:
        start=datetime.datetime.now(datetime.timezone.utc).isoformat();tick=time.time();print('Starting',args,flush=True)
        with (HERE/log).open('w') as output:result=subprocess.run([sys.executable,*args],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
        records.append(dict(command=[sys.executable,*args],log=log,started_utc=start,seconds=time.time()-tick,returncode=result.returncode))
        save(OUT/'pipeline_execution.json',dict(steps=records,passed=result.returncode==0 and len(records)==len(commands)))
        if result.returncode:raise RuntimeError(f'Pipeline stopped: {log}')
    print('Evaluation, figures and tests complete; report still requires research review.',flush=True)

if __name__=='__main__':main()
