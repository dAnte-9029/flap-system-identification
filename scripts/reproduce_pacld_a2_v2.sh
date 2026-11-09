#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:scripts
A2_PY=/home/zn/anaconda3/envs/flap-train-gpu/bin/python
"$A2_PY" scripts/diagnose_pacld_a2_v2.py
"$A2_PY" scripts/report_pacld_a2_v2.py
# Stage two is authorized by the committed stage-one conclusion and frozen protocol.
"$A2_PY" - <<'PY'
from diagnose_pacld_a2 import extract
from a2_raw_context import attach
from pathlib import Path
import pandas as pd
out=Path('experiments/phase_anchored_coupled_latent_v1/control_effectiveness_a2_v2/validation_features')
extract('validation',out)
attach(pd.read_pickle(out/'features.pkl'),'validation',out)
PY
"$A2_PY" scripts/evaluate_pacld_a2_v2_gpu.py --split train --resume
"$A2_PY" scripts/evaluate_pacld_a2_v2_gpu.py --split validation --resume
"$A2_PY" scripts/score_pacld_a2_v2.py
"$A2_PY" scripts/validation_gain_statistics_a2_v2.py
"$A2_PY" scripts/report_pacld_a2_v2_final.py
"$A2_PY" scripts/verify_pacld_a2_v2.py
