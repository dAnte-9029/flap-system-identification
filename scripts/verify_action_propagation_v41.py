"""Independent metric recomputation and native-API checks from saved states."""
import copy
import numpy as np
import train_control_windows_v2 as base
from run_action_propagation_v41 import OUT,PARENT,DATA,read,fold_data
from coupled_replay_v40 import VARIANTS,angles
from system_identification.control.px4_c9af import C9Controller


def restored(template,v,schema):
    snapshot=template.snapshot();offset=0
    for name,width in schema:
        values=v[offset:offset+width];offset+=width
        if width==1:value=float(values[0])
        elif name in ['positive','negative']:value=[bool(x) for x in values]
        else:value=values.tolist()
        snapshot['state'][name]=value
    c=copy.copy(template);c.restore(snapshot)
    return c


def main():
    assert read(OUT/'verification.json')['status']=='passed'
    protocol=read(OUT/'protocol.json');schema=protocol['state_schema'];branches=protocol['branches']
    with np.load(DATA/'windows.npz') as z:logged=z['input_future_command'].astype(float)
    metrics=0;native_calls=0
    for fold in range(5):
        ids,tq,tr,iq,ir,tape,air,manual,snaps=fold_data(fold)
        for variant,tick,feedback in VARIANTS:
            with np.load(OUT/f'fold{fold}/{variant}_teacher_states.npz') as z:ts=z['state']
            with np.load(PARENT/f'fold{fold}/{variant}_teacher.npz') as z:teacher=z['commands']
            templates=[C9Controller(snaps[i][variant]['parameters']) for i in ids]
            for stage in [30,35]:
                for seed in [17,23,42]:
                    root=OUT/f'fold{fold}';name=f'{variant}_v{stage}_seed{seed}'
                    with np.load(root/(name+'.npz')) as z:a={k:z[k] for k in z.files}
                    with np.load(PARENT/f'fold{fold}'/(name+'.npz')) as z:
                        mq=np.concatenate([iq[:,None],z['quaternion'][:,:49]],1)
                        mr=np.concatenate([ir[:,None],z['rates_b'][:,:49]],1)
                    with np.load(root/(name+'_metrics.npz')) as z:
                        for hi,h in enumerate([5,25,50]):
                            for ti,target in enumerate([logged[ids],teacher]):
                                expected=((a['commands'][:,:h]-target[:,:h,None,:])**2).sum(1)/h
                                np.testing.assert_allclose(z['mse'][:,hi,ti],expected,rtol=1e-12,atol=1e-12)
                                metrics+=expected.size
                    # Every parent/model/variant at fixed0,.48,.98s, all14branches.
                    for k in [0,24,49]:
                        ma=angles(mq[:,k]);ta=angles(tq[:,k])
                        for j,template in enumerate(templates):
                            cm=restored(template,a['model_controller_state'][j,k],schema)
                            ct=restored(template,ts[j,k],schema)
                            for bi,branch in enumerate(branches):
                                c=copy.copy(ct if branch in ['tm','tt'] else cm)
                                c.state=copy.deepcopy(c.state)
                                rp=(ta[j] if branch in ['mt','tt','attitude'] else ma[j]).copy()
                                rate=(tr[j,k] if branch in ['mt','tt','rates'] else mr[j,k]).copy()
                                if branch in ['p','q','r']:rate['pqr'.index(branch)]=tr[j,k,'pqr'.index(branch)]
                                if branch=='roll':rp[0]=ta[j,0]
                                if branch=='pitch':rp[1]=ta[j,1]
                                if branch=='integral':c.state.integral=ct.state.integral.copy()
                                if branch=='compression':
                                    for key in ['gain','cached_gain','hpf','energy','previous_effort']:setattr(c.state,key,copy.deepcopy(getattr(ct.state,key)))
                                if branch=='yaw_memory':
                                    c.state.yaw_euler=ct.state.yaw_euler;c.state.yaw_body=ct.state.yaw_body;c.state.last_rate_sp=ct.state.last_rate_sp.copy()
                                reference=tape[j,k*8]
                                sp=c.attitude(rp[0],rp[1],reference[0],reference[1],float(air[j]),0.,manual=bool(manual[j]),yaw_stick=reference[3])
                                d=c.rate(rate,sp,[0.,0.,0.],float(air[j]),0.,reference[2],tick*1e-6,
                                    landed=False,reset_integral=False,feedback=feedback)
                                np.testing.assert_array_equal(d['commands'],a['commands'][j,k,bi])
                                np.testing.assert_array_equal(np.abs(d['raw_surfaces'])>1.,a['surface_clipped'][j,k,bi])
                                native_calls+=1
    base.write(OUT/'independent_verification.json',dict(status='passed',all_window_metric_values_recomputed=metrics,
        saved_state_native_calls_verified=native_calls,checked_steps=[0,24,49],all14_branches_checked=True,
        snapshots_complete_restore_passed=True,commands_and_saturation_exact=True))
    print(dict(status='passed',metrics=metrics,native_calls=native_calls),flush=True)


if __name__=='__main__':main()
