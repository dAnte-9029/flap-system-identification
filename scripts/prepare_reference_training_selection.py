"""Choose a paired configuration cohort using coverage metadata, never errors."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from system_identification.data.reference_windows import extract_window

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/analysis/results/reference_dynamics_data_v1'
OUT=BASE/'training_ready'
CACHE=ROOT/'dataset/reference_dynamics_data_v1'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')


def run():
    OUT.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((CACHE/'manifest.json').read_text())
    for p,h in {**manifest['tables_sha256'],**manifest['cache_sha256']}.items():assert sha(ROOT/p)==h
    w=pd.read_csv(BASE/'windows.csv')
    coverage=w.groupby(['group','partition']).agg(windows=('window_id','size'),flights=('log_id','nunique')).unstack(fill_value=0)
    eligible=coverage[(coverage['flights']['train']>=2)&(coverage['flights']['validation']>=2)]
    assert len(eligible),'No paired config cohort; do not mix firmware or redraw splits'
    ranking=eligible.sort_values(('windows','train'),ascending=False)
    group=str(ranking.index[0]);selected=w[w.group==group].copy()
    write(OUT/'protocol.json',dict(parent_manifest_sha256=sha(CACHE/'manifest.json'),script_sha256=sha(__file__),
        rule='Require>=2 original train and>=2 original validation flights in same recorded firmware/config group, rank only by train window count; all modes and active/quiet references retained',
        selection_uses_validation='configuration and eligible flight presence only; never outputs/errors or validation normalization',
        reason='Largest train-only group has no original matching validation. Preserve original selection and use this explicitly separate paired-cohort proposal.',
        group=group,original_split_preserved=True,sealed_test_accessed=False,model_training_started=False))
    selected.to_csv(OUT/'windows.csv',index=False)
    state=[];reference=[];actuator=[];sources=[];checked=0
    for path,rows in selected.groupby('cache'):
        with np.load(ROOT/path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        assert rows.partition.nunique()==1
        split=rows.partition.iloc[0]
        for r in rows.itertuples():extract_window(a,int(r.origin_index));checked+=1
        sources.append(dict(log_id=rows.log_id.iloc[0],partition=split,cache=path,sha256=sha(ROOT/path)))
        if split=='train':
            hist=np.unique(np.concatenate([np.arange(i-25,i+1) for i in rows.origin_index]))
            refs=np.unique(np.concatenate([np.arange(i-25,i+50) for i in rows.origin_index]))
            state.append(a['state_features'][hist]);reference.append(a['references'][refs]);actuator.append(a['actuator_commands'][hist-1])
    stats={}
    for name,items in [('state',state),('reference',reference),('past_actuator',actuator)]:
        x=np.concatenate(items).astype(float);assert np.isfinite(x).all()
        stats[name+'_mean']=x.mean(0);stats[name+'_scale']=x.std(0).clip(1e-6)
    stats['state_mean'][9:11]=0;stats['state_scale'][9:11]=1
    norm=CACHE/'paired_normalization.npz';np.savez(norm,**stats)
    ref=np.concatenate(reference)
    pd.DataFrame(dict(feature=['roll_reference_rad','pitch_reference_rad','yaw_rate_reference','thrust_body_x'],
        minimum=ref.min(0),q01=np.quantile(ref,.01,axis=0),q99=np.quantile(ref,.99,axis=0),maximum=ref.max(0))).to_csv(OUT/'train_reference_ranges.csv',index=False)
    selected.groupby(['partition','mode','kind','active_reference']).size().rename('windows').reset_index().to_csv(OUT/'summary.csv',index=False)
    # Development CV is whole TRAIN flight only; original validation remains external.
    train=selected[selected.partition=='train'];flight_counts=train.groupby('log_id').size()
    folds={log:i for i,log in enumerate(sorted(flight_counts.index))}
    write(OUT/'train_leave_one_flight_out_folds.json',folds)
    write(OUT/'manifest.json',dict(group=group,sources=sources,parent_manifest=str((CACHE/'manifest.json').relative_to(ROOT)),
        parent_sha256=sha(CACHE/'manifest.json'),windows_path=str((OUT/'windows.csv').relative_to(ROOT)),windows_sha256=sha(OUT/'windows.csv'),
        normalization_path=str(norm.relative_to(ROOT)),normalization_sha256=sha(norm),
        normalization_fit_partitions=['train'],phase_contract=manifest['phase_contract'],frequency_contract=manifest['frequency_contract'],
        checked_windows=checked,train_windows=len(train),validation_windows=int((selected.partition=='validation').sum()),
        train_flights=len(flight_counts),validation_flights=selected[selected.partition=='validation'].log_id.nunique(),
        history_steps=26,forecast_steps=50,dt_s=.02,formal_training_started=False,control_validated=False,
        cv_warning='Fold-specific scalers must be re-fit inside each train-only fold. Full-train scaler is for final all-train fit and original validation only.'))
    write(OUT/'completion.json',dict(status='ready_for_training_preflight',outputs={p.name:sha(p) for p in OUT.iterdir() if p.is_file()},
        normalization_sha256=sha(norm),formal_training_started=False,sealed_test_accessed=False))
    print(json.dumps(json.loads((OUT/'manifest.json').read_text()),indent=2,ensure_ascii=False))


if __name__=='__main__':run()
