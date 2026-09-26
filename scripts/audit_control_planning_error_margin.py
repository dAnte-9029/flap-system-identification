"""Evaluate frozen planning gains against a train-derived empirical error allowance."""
from pathlib import Path
import numpy as np
import pandas as pd
import run_paper_rollout_horizon_ablation as source
from system_identification.integration.control_error_envelope import EmpiricalErrorEnvelope

m=source.m
ROOT=m.ROOT/'docs/analysis/results/control_dynamics_realdata_v1'
OUT=ROOT/'planning_error_margin'


def run():
    OUT.mkdir(parents=True,exist_ok=False)
    for exp in ['past_residual_information','planning_stress']:
        m.verify_pins(m.read_json(ROOT/exp/'completion.json')['outputs'])
    bounds=ROOT/'past_residual_information/empirical_envelope.csv'
    inputs=ROOT/'planning_stress/per_origin.csv'
    pins={m.rel(p):m.file_hash(p) for p in [Path(__file__),bounds,inputs,m.ROOT/'src/system_identification/integration/control_error_envelope.py']}
    m.write_json(OUT/'protocol.json',dict(pins=pins,heldout_accessed=False,
        rule='nominal reduction in absolute q target error must exceed2*fixed train empirical q error threshold',
        inequality='triangle inequality conditional on BOTH forecast errors bounded by B; those errors need not be independent',
        caveat='counterfactual bound assumptions not validated; only an empirical stress gate, not certified action safety'))
    envelope=EmpiricalErrorEnvelope(pd.read_csv(bounds).to_dict('records'))
    f=pd.read_csv(inputs);f=f[f.destination=='ensemble'].copy()
    # Error magnitudes are sufficient for this comparison; no need to reconstruct signed forecasts.
    allowance=np.array([2*envelope.bound(int(k),'q') for k in f.steps])
    f['empirical_allowance']=allowance
    f['nominal_improvement']=f.hold_abs_error-f.selected_abs_error
    f['margin']=f.nominal_improvement-allowance
    f['passes_empirical_gate']=f.assessable&(f.margin>0)
    f.to_csv(OUT/'per_origin.csv',index=False)
    f.groupby(['policy','steps','offset']).agg(requested=('origin','size'),assessable=('assessable','sum'),
        passes=('passes_empirical_gate','sum'),max_nominal_improvement=('nominal_improvement','max')).to_csv(OUT/'summary.csv')
    m.verify_pins(pins)
    m.write_json(OUT/'completion.json',dict(status='complete',heldout_accessed=False,
        assessed=int(f.assessable.sum()),passed=int(f.passes_empirical_gate.sum()),control_validated=False,
        outputs={m.rel(p):m.file_hash(p) for p in OUT.iterdir() if p.is_file()}))


if __name__=='__main__':run()
