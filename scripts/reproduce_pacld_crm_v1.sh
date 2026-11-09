#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:scripts
export MPLCONFIGDIR=/tmp/pacld-crm-matplotlib
CRM_PY=/home/zn/anaconda3/envs/flap-train-gpu/bin/python
"$CRM_PY" scripts/pacld_crm_v1.py
"$CRM_PY" scripts/pacld_crm_gradients.py --microbatch 4
"$CRM_PY" scripts/pacld_crm_report.py
"$CRM_PY" scripts/verify_pacld_crm_v1.py
