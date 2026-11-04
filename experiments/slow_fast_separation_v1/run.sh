#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=/home/zn/anaconda3/envs/flap-train-gpu/bin/python
OUT=${1:-docs/analysis/results/slow_fast_separation_v1/reproduction}
"$PY" -m pytest -q experiments/slow_fast_separation_v1/test_methods.py
"$PY" experiments/slow_fast_separation_v1/run.py --output "$OUT"
"$PY" experiments/slow_fast_separation_v1/summarize.py "$OUT"
