"""Raw-log target recomputation and equal-flight teacher-replay summaries."""
import numpy as np
import pandas as pd
from pyulog import ULog
import train_control_windows_v2 as base
from run_c9af_replay_v38 import OUT,PREV,DATA,VARIANTS,read
from audit_controller_support_v26 import verify_parent
from system_identification.control.px4_c9af import C9Controller


def main():
    assert read(OUT/'status.json')['status']=='replay_complete_evaluation_pending'
    windows=pd.read_csv(DATA/'windows.csv');all_rows=[];cover=[];recomputed=0;restores=0
    for result in read(OUT/'status.json')['completed']:
        dest=OUT/result['destination'];m=pd.read_csv(dest/'metrics.csv');all_rows.append(m);cover.append(pd.read_csv(dest/'coverage.csv'))
        entry=next(e for e in read(PREV/'dataset_manifest.json')['sources'] if e['log_id']==result['log_id'])
        assert base.sha(entry['source']['path'])==entry['source']['sha256']
        u=ULog(entry['source']['path'],message_name_filter_list=['vehicle_rates_setpoint','vehicle_torque_setpoint','actuator_motors','actuator_servos'])
        ds={d.name:d.data for d in u.data_list if d.multi_id==0}
        with np.load(base.ROOT/entry['cache']) as z:times=z['timestamp_us']
        arrays=np.load(dest/'predictions.npz');indexed=m.set_index(['array_index','variant','branch','kind','axis']).sort_index()
        for parent,g in windows[windows.log_id==result['log_id']].groupby('array_index'):
            origin=int(times[g.origin_index.item()]);query=origin+np.arange(50)*20000
            target={}
            for name,fields in [('vehicle_rates_setpoint',['roll','pitch','yaw']),('vehicle_torque_setpoint',[f'xyz[{i}]' for i in range(3)]),
                                ('actuator_motors',['control[0]']),('actuator_servos',[f'control[{i}]' for i in range(3)])]:
                d=ds[name];idx=[]
                for t in query:
                    allowed=np.flatnonzero((d['timestamp']<=t)&(d.get('timestamp_sample',d['timestamp'])<=t))
                    idx.append(int(allowed[np.argmax(d['timestamp'][allowed])]))
                target[name]=np.column_stack([d[k][idx] for k in fields]).astype(float)
            truth=dict(rates=target['vehicle_rates_setpoint'],torque=target['vehicle_torque_setpoint'],commands=np.column_stack([target['actuator_motors'],target['actuator_servos']]))
            for i in np.flatnonzero(arrays['array_index']==parent):
                name=str(arrays['variant'][i]);branch=str(arrays['branch'][i])
                for kind,t in truth.items():
                    error=arrays[kind][i]-t
                    rows=indexed.loc[(parent,name,branch,kind)]
                    np.testing.assert_allclose(rows.mse.to_numpy(),np.mean(error**2,axis=0),rtol=1e-10,atol=1e-12)
                    np.testing.assert_allclose(rows.bias.to_numpy(),np.mean(error,axis=0),rtol=1e-10,atol=1e-12)
                    recomputed+=1
        states=read(dest/'initial_states.json')
        # Exact restore of every recorded partial-initialization hypothesis.
        for item in states:
            value=item['initial'];c=C9Controller(value['parameters']);c.restore(value);assert c.snapshot()==value;restores+=1
        arrays.close()
    m=pd.concat(all_rows,ignore_index=True);coverage=pd.concat(cover,ignore_index=True)
    for _,g in m[(m.kind=='commands')&(m.axis==0)].groupby(['variant','branch']):np.testing.assert_array_equal(np.sort(g.array_index),np.arange(407))
    m.to_csv(OUT/'metrics_per_window.csv',index=False);coverage.to_csv(OUT/'coverage.csv',index=False)
    flight=m.groupby(['variant','branch','kind','axis','log_id']).mse.mean().pow(.5).reset_index(name='rmse')
    flight.to_csv(OUT/'metrics_per_flight.csv',index=False)
    flight.groupby(['variant','branch','kind','axis']).rmse.mean().to_csv(OUT/'summary_rmse.csv')
    # Split by original mode, with counts; keep both Mission andStabilized.
    merged=m.merge(windows[['array_index','mode']],on='array_index',validate='many_to_one')
    mode=merged.groupby(['variant','branch','kind','axis','mode','log_id']).mse.mean().pow(.5).reset_index(name='rmse')
    mode.to_csv(OUT/'by_mode_flight.csv',index=False)
    for name,digest in read(OUT/'protocol.json')['code'].items():assert base.sha(base.ROOT/name)==digest
    verify_parent(PREV);verify_parent(DATA)
    base.write(OUT/'verification.json',dict(status='passed',all407_in_every_branch_and_variant=True,variants=7,branches=2,
        raw_target_metric_groups_recomputed=recomputed,serialized_initial_hypotheses_restored=restores,
        initial_hypotheses_not_exact_physical_state=True,unsupported_scored_cycles=int(coverage.unsupported_future_cycles.sum()),
        no_future_hidden_state_injection=True,model_replay_started=False,validation_accessed=False,sealed_test_accessed=False,reserved_accessed=False))
    print('v38 target and metric verification complete',flush=True)


if __name__=='__main__':main()
