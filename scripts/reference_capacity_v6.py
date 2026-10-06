"""Common immutable-data contracts for budget, output and predictability checks."""
import json
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from system_identification.models.reference_gru_unit_gravity import ReferenceGRUUnitGravity
from system_identification.models.reference_gru_direct_rate import ReferenceGRUDirectRate
from evaluate_control_windows_v2 import error_rows,summarize
OUT=base.ROOT/'docs/analysis/results/control_windows_v6'
CLASSES={'integrated':ReferenceGRUUnitGravity,'direct':ReferenceGRUDirectRate}


def model_from(path):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck['feature_contract']!='unit_gravity_v1':raise ValueError('wrong feature contract')
    names=['state_mean','state_scale','reference_mean','reference_scale','actuator_mean','actuator_scale','rate_scale','velocity_scale']
    m=CLASSES[ck['arm']]({k:ck['state_dict'][k] for k in names});m.load_state_dict(ck['state_dict']);return m.eval()


def metric_frames(pred,truth,meta,arm,seed):
    f=summarize(error_rows(pred,truth,meta,arm,seed))
    wave=meta[['array_index','window_id','log_id','partition']].copy();wave['arm']=arm;wave['seed']=seed
    for j,name in enumerate(['p','q','r']):wave[name]=np.sqrt(np.mean((pred['rates_b'][:,:,j]-truth['rates_b'][:,:,j])**2,axis=1))
    wave['velocity']=np.sqrt(np.mean((pred['velocity_n']-truth['velocity_n'])**2,axis=(1,2)))
    dot=abs(np.sum(pred['quaternion']*truth['quaternion'],axis=-1)).clip(0,1);wave['attitude']=np.sqrt(np.mean((2*np.arccos(dot))**2,axis=1))
    return f,wave


def freeze():
    OUT.mkdir(exist_ok=False);x,y,stats=base.load()
    parent=base.ROOT/'docs/analysis/results/control_windows_v4';old=json.loads((parent/'protocol.json').read_text())
    paths=[base.DATA/'manifest.json',base.ROOT/'dataset/control_windows_v2/windows.npz',base.ROOT/'dataset/control_windows_v2/normalization.npz',
        parent/'tiny_fit.pt',base.ROOT/'src/system_identification/models/reference_gru_unit_gravity.py',base.ROOT/'src/system_identification/models/reference_gru_direct_rate.py']
    protocol=dict(source_pins={str(p.relative_to(base.ROOT)):base.sha(p) for p in paths},
        data='same original50Hz H26/K50,407train739developmentvalidation; no selection/filter change',
        model='same40070parameters, initial weights and normalization; direct mode changes rate parametrization to origin+6sigma-bounded direct residual; velocity/quaternion update unchanged',
        output_bound='direct rate residual clipped to+/-6 original train rate scales, versus integrated+/-120rad/s2 increments; this bound change belongs to the output parametrization and must be reported',
        tiny_cohort=old['tiny_cohort'],tiny_seed=17,tiny_batch=8,snapshots=[1600,6400],
        budget='rerun from original seed with optimizer state preserved through6400; verify integrated1600 reproduces frozen tiny/whole-training checkpoint; save optimizer for future work',
        optimizer='AdamW3e-4wd1e-4,clip1,unchanged original raw rate+velocity+0.2attitude loss',
        tiny_gates='budget: integrated6400 vs integrated1600; output: direct1600 vs integrated1600 AND direct6400 vs integrated6400. Each requires>=15percent fit8 p/q full1s waveform improvement, no attitude/velocity >10percent regression. Uses train only.',
        formal_policy='if budget gate passes, integrated6400 x3seeds. If either direct gate passes, BOTH integrated and direct6400 x3seeds, saving1600 as prespecified paired checkpoint. No val-dependent run/checkpoint choices.',
        formal_seeds=[17,23,42],formal_batch=32,formal_primary='activep/q at0.5/1s, equal flights then axes/horizons/seeds; budget6400-v1600 and output direct-vintegrated at fixed1600/6400 separately',
        promotion='>=5percent primary improvement everyseedpositive and no allmetric>5percent regression; absolute0.5s activep/q<=.15rad/s attitude<=3deg velocity<=.5m/s eachseed; independent control-response evidence remains separate',
        predictability=dict(harmonic_orders=[1,2,3,4,6],harmonic_ridge=.01,history='26 observed samples; fit past only; intercept retained, no forced future anchoring',
            phase='held-frequency operational forecast vs future-logged-phase diagnostic only; in-sample future projection descriptive, not a predictive bound',
            linear_ridge_alpha=10,linear_validation='five-train-flight leave-one-flight-out plus fixed development validation',horizons_steps=[1,5,25,50],
            conclusion_limit='failure of finite baselines does not prove irreducible noise; no repeated identical-condition experiment to certify noise floor'),
        validation_status='already viewed development set, not blind test',sealed_test_accessed=False)
    base.write(OUT/'protocol.json',protocol)
    delta=(y['rates_b']-x['initial_rates_b'][:,None,:]).numpy()/stats['rate_scale']
    w=pd.read_csv(base.DATA/'windows_v2.csv');rows=[]
    for part in ['train','validation']:
        d=abs(delta[w.partition.to_numpy()==part])
        rows.append(dict(partition=part,above_direct_6sigma_fraction=float(np.mean(d>6)),maximum_normalized_residual=float(d.max())))
    base.write(OUT/'output_bound_audit.json',dict(bounds=rows,gate_changed=False))
if __name__=='__main__':freeze()
