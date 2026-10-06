"""Freeze one bounded loss-weighting experiment before training or evaluation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import train_control_windows_v2 as base
from reference_groupcv_v8 import load_train
from reference_frequency_weight_v13 import frequency_weights
from run_reference_generalization_v7 import verify_manifest
OUT=base.ROOT/'docs/analysis/results/control_windows_v13'
V8=OUT.parent/'control_windows_v8';V12=OUT.parent/'control_windows_v12'


def main():
    OUT.mkdir(exist_ok=False)
    for source in [V8,V12]:verify_manifest(source/'delivery_manifest.json',['artifacts','code'])
    x,y,w=load_train();m=json.loads((base.DATA/'manifest.json').read_text())
    paths=[V8/'delivery_manifest.json',V12/'delivery_manifest.json',base.DATA/'manifest.json',base.DATA/'windows_v2.csv',base.ROOT/'dataset/control_windows_v2/windows.npz',
        base.ROOT/'scripts/reference_frequency_weight_v13.py',Path(__file__),base.ROOT/'scripts/train_reference_frequency_weight_v13.py',base.ROOT/'scripts/run_reference_frequency_weight_v13.py',
        base.ROOT/'scripts/evaluate_reference_frequency_weight_v13.py',base.ROOT/'tests/test_reference_frequency_weight_v13.py']
    base.write(OUT/'protocol.json',dict(stage='v13 bounded history-frequency q-loss balancing',source_pins={str(p.relative_to(base.ROOT)):base.sha(p) for p in paths},
        phase_contract=m['phase_contract'],frequency_contract=m['frequency_contract'],data='407train windows, original5outerflight folds; no data removal, new split, development or sealed/reserved evaluation',
        model='same v8hold ReferenceGRUPhaseSchedule,40,070params,phase_start=None; holdfrequency; H26/K50 at50Hz; no yawmean correction',
        reproduction='one extra fold0seed17 training with weights1; parameter match to oldv8<=1e-6 required',runs='5folds xseeds17/23/42;1600updates,batch32,AdamW3e-4wd1e-4clip1; freshweights, identical v8 initialparameter hashes and saved sampledwindow tapes; no early stopping',
        weights='one axis q only; historymean frequency. Fit global flight-balanced terciles on4fitflights (not separate equal-count terciles perflight, which would be a near-noop). For eachflight and occupiedbin raw=n_flight/(occupied_bins*n_bin). Find scalarc with mean(clip(c*raw,.5,2))=1. Empty bins not invented. Allwindowspositive.',
        objective='oldloss + mean((weight-1)*mean_time((qpred-qtruth)/qscale)^2)/3; no minibatchweight renormalization, p/r/velocity/attitude direct losses unchanged; sharednetwork can change all predictions',
        primary='qfullwave RMSE, equalflights then horizons.5/1s then seeds; compare unweighted common windows against frozenv8hold',
        gate='primarygain>=5%;all3seedsgain>0;>=4/5flightgain>0; eachseed allstateendpoints .5/1s regression<=5%; p/rfullwave eachseed/horizonregression<=5%; qOOF low/mid/high eachseed/horizonregression<=5%. q1s referenceutility loss<=.02 absolute perseed and referenceeffect RMS ratio in[.9,1.1].',
        reference='recordedreference vs all5heldatstart; utility=1-factualRMSE/heldRMSE againstsameobservedtruth; not causal counterfactual flightresponse',
        absolute='eachseed allstate.5s p/q/r<=.15rad/s,attitude<=3deg,velocity<=.5m/s; independentresponse/closedloop stillrequired',
        selection='singlecandidate chosen afterv12exploration; reusedOOFflights are development evidence, no multiplecomparison-adjusted claim',
        bootstrap='2000 pairedwholeflight resamples seed913; exploratory5flights',
        after_gate='report complete OOF result and freeze decision; any all407 refit/development evaluation is a separate subsequent stage, no auto expansion in this run',sealed_test_accessed=False))
    allrecords=[]
    for fold in range(5):
        source=V8/f'fold{fold}';info=json.loads((source/'manifest.json').read_text());fit=np.array(info['fit_indices']);query=np.array(info['query_indices']);dest=OUT/f'fold{fold}';dest.mkdir()
        weight,cuts,labels,records=frequency_weights(x['history_state'][fit,:,11].numpy().mean(1,dtype=float),w.iloc[fit].log_id.to_numpy())
        np.savez_compressed(dest/'weights.npz',fit_indices=fit,query_indices=query,weight=weight,cuts=cuts,stratum=labels)
        base.write(dest/'manifest.json',dict(fit_indices=fit.tolist(),query_indices=query.tolist(),source_fold_sha256=base.sha(source/'manifest.json'),weights_sha256=base.sha(dest/'weights.npz')))
        allrecords.extend(dict(fold=fold,**r) for r in records)
    pd.DataFrame(allrecords).to_csv(OUT/'weight_support.csv',index=False)

if __name__=='__main__':main()
