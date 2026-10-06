"""Report mined natural references without treating them as randomized inputs."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/analysis/results/existing_reference_ramps_v1'


def run():
    for name in ['existing_reference_excitation_v1','existing_reference_ramps_v1']:
        base=BASE.parent/name
        receipt=json.loads((base/'completion.json').read_text())
        for relative, expected in receipt['outputs'].items():
            assert hashlib.sha256((base/relative).read_bytes()).hexdigest()==expected
    f=pd.read_csv(BASE/'candidates.csv');primary=f[f.threshold_deg==1].copy()
    b=pd.read_csv(BASE/'blocks.csv');logs=pd.read_csv(BASE/'logs.csv')
    primary['sign']=np.where(primary.amplitude_deg>0,'positive','negative')
    summary=primary.groupby(['partition','axis_name']).agg(candidates=('candidate_id','size'),flights=('log_id','nunique'),single_axis=('single_axis','sum'))
    summary.to_csv(BASE/'primary_summary.csv')
    groups=logs.groupby(['firmware','parameter_hash','partition']).size().unstack(fill_value=0)
    groups.to_csv(BASE/'configuration_groups.csv')
    # Illustrations are first by log/time among reference-selected primary candidates.
    # No response score controls selection.
    picks=[]
    for axis in ['roll','pitch']:
        for sign in ['positive','negative']:
            take=primary[(primary.axis_name==axis)&(primary.sign==sign)].sort_values(['log_id','onset_timestamp_us'])
            if len(take):picks.append(take.iloc[0])
    fig,axes=plt.subplots(4,len(picks),figsize=(15,10),sharex='col',squeeze=False)
    for col,row in enumerate(picks):
        with np.load(BASE/'segments'/f'{row.candidate_id}.npz') as z:
            t=z['relative_time_s'];axis=int(row.axis)
            axes[0,col].plot(t,z['reference_deg'][:,axis],label='reference')
            axes[0,col].plot(t,z['attitude_deg'][:,axis],label='observed')
            axes[0,col].set_title(f'{row.axis_name} {row.amplitude_deg:+.2f} deg\n{row.candidate_id}',fontsize=9)
            axes[1,col].plot(t,z['reference_deg'][:,1-axis],label='other reference',color='tab:orange')
            for i in range(3):axes[2,col].plot(t,z['servo_commands'][:,i],label=f'servo {i}')
            axes[3,col].plot(t,z['rates_rad_s'][:,axis],color='tab:green')
            for a in axes[:,col]:a.axvline(0,color='gray',ls='--');a.grid(alpha=.2)
            axes[3,col].set_xlabel('Time relative to candidate boundary (s)')
    for a,label in zip(axes[:,0],['Axis angle (deg)','Other reference (deg)','Normalized command','Body rate (rad/s)']):a.set_ylabel(label)
    axes[0,0].legend();axes[2,0].legend(fontsize=8)
    fig.suptitle('Natural reference candidates: observational examples, not controlled experiments')
    fig.tight_layout();fig.savefig(BASE/'examples.png',dpi=140);plt.close(fig)
    selected=[dict(candidate_id=r.candidate_id,log_id=r.log_id,seconds_from_log_start=float(r.seconds_from_log_start)) for r in picks]
    (BASE/'example_selection.json').write_text(json.dumps(selected,indent=2,ensure_ascii=False)+'\n')
    print(summary.to_string())
    print('fresh primary',int((primary.chain_fresh_fraction>=.99).sum()),'of',len(primary))
    print('fresh blocks',int((b.chain_fresh_fraction>=.99).sum()),'of',len(b))
    print('same configuration groups on both splits',int(((groups.train>0)&(groups.validation>0)).sum()))


if __name__=='__main__':run()
