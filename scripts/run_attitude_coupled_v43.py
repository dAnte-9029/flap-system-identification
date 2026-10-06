"""Reuse frozen v40 causal conditions and unmodified coupling for v43 models."""
import concurrent.futures
import numpy as np
import torch
import train_control_windows_v2 as base
from train_attitude_v42 import OUT,PREP,DATA,read
from evaluate_attitude_v43 import V40,SEEDS
from actuator_training_v30 import model_from
from coupled_replay_v40 import VARIANTS,rollout
from audit_controller_support_v26 import verify_parent


def worker(fold):
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    info=read(DATA/f'fold{fold}/manifest.json');ids=info['query_indices']
    with np.load(DATA/'windows.npz') as z:
        history={k[6:]:torch.tensor(z[k][ids]) for k in z.files if k.startswith('input_') and k not in ['input_future_command','input_dt_s']}
    bounds=read(V40/f'fold{fold}/support.json');assert bounds['query_indices']==ids and bounds['fit_indices']==info['fit_indices']
    support={k:np.asarray(bounds[k]) for k in ['command_min','command_max','state_min','state_max']}
    matches=[]
    for e in read(V40/'conditions_manifest.json')['flights']:
        with np.load(V40/e['path']/'conditions.npz') as z:
            if set(z['array_index'])==set(ids):matches.append(e)
    assert len(matches)==1;entry=matches[0];source=V40/entry['path']
    for n,d in entry['artifacts'].items():assert base.sha(source/n)==d
    with np.load(source/'conditions.npz') as z:
        order=[list(z['array_index']).index(i) for i in ids]
        tape=z['tape'][order];air=z['air'][order];manual=z['manual'][order]
    snapshots={s['array_index']:s['states'] for s in read(source/'snapshots.json')}
    dest=OUT/f'coupled/fold{fold}';dest.mkdir(exist_ok=False);count=0
    for variant in VARIANTS:
        name=variant[0];snap=[snapshots[i][name] for i in ids]
        for seed in SEEDS:
            path=OUT/f'fold{fold}/seed{seed}/final.pt'
            assert base.sha(path)==read(path.parent/'status.json')['checkpoint_sha256']
            model=model_from(path)
            with np.load(DATA/f'fold{fold}/normalization.npz') as z:
                for k in z.files:np.testing.assert_array_equal(getattr(model,k).numpy(),z[k].astype(np.float32))
            result=rollout(model,history,snap,tape,air,manual,variant,support)
            assert all(np.isfinite(v).all() for v in result.values())
            with np.load(V40/f'fold{fold}/{name}_v35_seed{seed}.npz') as z:
                np.testing.assert_array_equal(result['commands'][:,0],z['commands'][:,0])
            np.savez_compressed(dest/f'{name}_seed{seed}.npz',array_index=ids,**result)
            count+=1;base.write(dest/'status.json',dict(status='running',completed=count,total=12))
        print(f'fold{fold} {name}: all3 candidates complete',flush=True)
    base.write(dest/'status.json',dict(status='complete',completed=count,total=12))
    return fold


def main():
    assert read(OUT/'conditional/verification.json')['status']=='passed'
    for p in [PREP,DATA,V40]:verify_parent(p)
    dest=OUT/'coupled';dest.mkdir(exist_ok=False)
    code=['scripts/run_attitude_coupled_v43.py','scripts/coupled_replay_v40.py']
    base.write(dest/'protocol.json',dict(preparation_sha256=base.sha(PREP/'delivery_manifest.json'),
        conditions_parent_sha256=base.sha(V40/'delivery_manifest.json'),variants=VARIANTS,
        code={n:base.sha(base.ROOT/n) for n in code},
        conditions='Exact v40 causal warmup, reference tapes, fixedCAS, model20ms first-command hold, native allocation; future plant/log actions absent.',
        baseline='Reuse all frozen v35 trajectories and matched teacher, no controller tuning or new clamps.'))
    base.write(dest/'status.json',dict(status='running'))
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as pool:list(pool.map(worker,range(5)))
    for p in [PREP,DATA,V40]:verify_parent(p)
    base.write(dest/'status.json',dict(status='replay_complete_evaluation_pending',trajectories=407*3*4))


if __name__=='__main__':main()
