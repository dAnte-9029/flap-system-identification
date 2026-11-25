#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
py=/home/zn/anaconda3/envs/flap-train-gpu/bin/python
"$py" export_predictions.py best
"$py" export_predictions.py terminal
"$py" compare.py
