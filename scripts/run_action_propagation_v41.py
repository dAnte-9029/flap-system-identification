"""Reconstruct frozen v40 histories and probe one-tick state substitutions."""
import concurrent.futures
import numpy as np
import torch
import train_control_windows_v2 as base
from run_coupled_replay_v40 import DATA,read
from coupled_replay_v40 import VARIANTS,controllers,interval,angles
from action_propagation_v41 import BRANCHES,SCHEMA,vector,from_vector,probes
from audit_controller_support_v26 import verify_parent
from system_identification.models.trajectory_main_v1 import _normalize_quaternion

OUT=base.ROOT/'docs/analysis/results/control_windows_v41'
PARENT=OUT.parent/'control_windows_v40'
PARENTS=[DATA,PARENT,OUT.parent/'control_windows_v39']


def fold_data(fold):
    ids=read(DATA/f'fold{fold}/manifest.json')['query_indices']
    with np.load(DATA/'windows.npz') as z:
        teacher_q=np.concatenate([z['input_initial_quaternion'][ids,None],z['label_quaternion'][ids,:49]],1)
        teacher_r=np.concatenate([z['input_initial_rates_b'][ids,None],z['label_rates_b'][ids,:49]],1)
        initial_q=_normalize_quaternion(torch.tensor(z['input_initial_quaternion'][ids])).numpy()
        initial_r=z['input_initial_rates_b'][ids]
    match=[]
    for e in read(PARENT/'conditions_manifest.json')['flights']:
        with np.load(PARENT/e['path']/'conditions.npz') as z:
            if set(z['array_index'])==set(ids):
                order=[list(z['array_index']).index(i) for i in ids]
                match=[e,z['tape'][order],z['air'][order],z['manual'][order]]
    e,tape,air,manual=match
    snaps={s['array_index']:s['states'] for s in read(PARENT/e['path']/'snapshots.json')}
    return ids,teacher_q,teacher_r,initial_q,initial_r,tape,air,manual,snaps


def worker(fold):
    torch.set_num_threads(1)
    p=read(OUT/'protocol.json')
    for name,digest in p['code'].items():assert base.sha(base.ROOT/name)==digest
    ids,tq,tr,iq,ir,tape,air,manual,snaps=fold_data(fold)
    dest=OUT/f'fold{fold}';dest.mkdir();runs=0;observations=0
    for variant,tick,feedback in VARIANTS:
        initial=[snaps[i][variant] for i in ids];templates=controllers(initial)
        tc=controllers(initial);teacher_states=[]
        with np.load(PARENT/f'fold{fold}/{variant}_teacher.npz') as z:teacher={k:z[k] for k in z.files}
        for k in range(50):
            teacher_states.append(np.stack([vector(c) for c in tc]))
            d=interval(tc,tq[:,k],tr[:,k],tape[:,k*8:(k+1)*8],air,manual,tick,feedback)
            for key in ['command','torque','rates_sp','micro_commands','surface_clipped']:
                np.testing.assert_array_equal(d[key],teacher['commands' if key=='command' else key][:,k])
        teacher_states=np.stack(teacher_states,1)
        np.savez_compressed(dest/f'{variant}_teacher_states.npz',array_index=ids,state=teacher_states)
        for stage in [30,35]:
            for seed in [17,23,42]:
                source=PARENT/f'fold{fold}/{variant}_v{stage}_seed{seed}.npz'
                with np.load(source) as z:saved={k:z[k] for k in z.files}
                np.testing.assert_array_equal(saved['array_index'],ids)
                mq=np.concatenate([iq[:,None],saved['quaternion'][:,:49]],1)
                mr=np.concatenate([ir[:,None],saved['rates_b'][:,:49]],1)
                mc=controllers(initial);commands=[];clipped=[];states=[]
                for k in range(50):
                    state=np.stack([vector(c) for c in mc]);states.append(state)
                    rp=angles(mq[:,k]);tp=angles(tq[:,k]);cc=[];ss=[]
                    for j,c in enumerate(mc):
                        t=from_vector(templates[j],teacher_states[j,k])
                        command,sat=probes(c,t,rp[j],mr[j,k],tp[j],tr[j,k],tape[j,k*8],air[j],manual[j],tick,feedback)
                        cc.append(command);ss.append(sat)
                    cc=np.stack(cc);ss=np.stack(ss)
                    # The probes must not change either reconstructed factual history.
                    np.testing.assert_array_equal(state,np.stack([vector(c) for c in mc]))
                    np.testing.assert_array_equal(cc[:,0],saved['commands'][:,k])
                    np.testing.assert_array_equal(cc[:,3],teacher['commands'][:,k])
                    commands.append(cc);clipped.append(ss)
                    d=interval(mc,mq[:,k],mr[:,k],tape[:,k*8:(k+1)*8],air,manual,tick,feedback)
                    for key in ['command','torque','rates_sp','micro_commands','surface_clipped']:
                        np.testing.assert_array_equal(d[key],saved['commands' if key=='command' else key][:,k])
                    observations+=len(ids)
                np.savez_compressed(dest/f'{variant}_v{stage}_seed{seed}.npz',array_index=ids,
                    commands=np.stack(commands,1),surface_clipped=np.stack(clipped,1),model_controller_state=np.stack(states,1))
                runs+=1
                base.write(dest/'status.json',dict(status='running',completed=runs,total=24))
        print(f'fold{fold} {variant}: factual replay exact, all14 probes complete',flush=True)
    base.write(dest/'status.json',dict(status='complete',completed=runs,total=24,
        observations=observations,all_windows=len(ids),all_factual_actions_torque_refs_micro_and_limits_exact=True))
    return fold


def main():
    for root in PARENTS:verify_parent(root)
    OUT.mkdir(exist_ok=False)
    code=['scripts/action_propagation_v41.py','scripts/run_action_propagation_v41.py','tests/test_action_propagation_v41.py']
    protocol=dict(stage='v41 frozen state-to-action error propagation',status='frozen_before_execution',
        code={f:base.sha(base.ROOT/f) for f in code},parents={str(p.relative_to(base.ROOT)):base.sha(p/'delivery_manifest.json') for p in PARENTS},
        scope='All407 originaltrain parents, all30model trajectories and4v40variants. No retraining, new controller design, window removal or developmentvalidation/sealed/reserved.',
        branches=BRANCHES,state_schema=SCHEMA,
        branch_semantics=dict(mm='model controller history/model current plant',mt='model history/measured plant',
            tm='teacher history/model plant',tt='teacher history/measured plant',
            p='Replace only measuredp at fixed model history',q='Replace only measuredq',r='Replace only measuredr',
            roll='Replace only measuredroll angle',pitch='Replace only measuredpitch angle',rates='Replace all three rates',
            attitude='Replace both roll/pitch angles',integral='Replace only model integral with teacher integral at model plant',
            compression='Replace gain/cachedgain/HPF/energy/previous-effort group with teacher; only cachedgain affects this tick output',
            yaw_memory='Replace oldyaw/bodyyaw/last-rate-sp memory with teacher at model plant'),
        timing='At each original20ms boundary before the attitude update, clone factual controller states and evaluate one native tick. Only factual histories advance all original microticks on unchanged v40 trajectories. No oracle is propagated into later state or GRU.',
        targets='Report each branch against real logged actions AND matchedteacher; teacher is approximate, not true firmware hidden state.',
        decomposition='D=mm-tt; P=mm-mt; H=mt-tt; exactD=P+H. Also reversed-order P2=tm-tt,H2=mm-tm; interaction=mm-mt-tm+tt. Signed cross moments of P,H withD are descriptive and may cancel; not causal percentages or additive independent effects.',
        evaluation='All axes, horizons .1/.5/1s prefixes, time curves, flight/mode and original v40 currentstate/command-support strata. Equalflight then equalseed RMSE. No new promotion thresholds, no best-oracle selection for deployment.',
        tolerance=dict(factual_rtol=0,factual_atol=0,identity_atol=1e-12),
        model_promoted=False,validation_accessed=False,sealed_test_accessed=False,reserved_accessed=False)
    base.write(OUT/'protocol.json',protocol)
    base.write(OUT/'dataset_manifest.json',dict(authority=str((PARENT/'dataset_manifest.json').relative_to(base.ROOT)),
        authority_sha256=base.sha(PARENT/'dataset_manifest.json'),contract=read(PARENT/'dataset_manifest.json'),
        future_measured_state='offline oracle probes and scoring only; not propagated',partitions=['original_train_only']))
    base.write(OUT/'status.json',dict(status='running',completed_folds=[]))
    completed=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as pool:
        for fold in pool.map(worker,range(5)):
            completed.append(fold);base.write(OUT/'status.json',dict(status='running',completed_folds=completed))
    for root in PARENTS:verify_parent(root)
    base.write(OUT/'status.json',dict(status='probes_complete_evaluation_pending',completed_folds=completed))


if __name__=='__main__':main()
