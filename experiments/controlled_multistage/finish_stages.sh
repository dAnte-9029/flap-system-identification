#!/usr/bin/env bash
set -euo pipefail
stage_src=$(cd "$(dirname "$0")" && pwd)
cd "$stage_src"
stage_out=/home/zn/flap-system-identification/outputs/controlled_multistage_20261120
stage_py=/home/zn/anaconda3/envs/flap-train-gpu/bin/python
names=(s2a s2b s2c s3a s3b s3c)
"$stage_py" -u evaluate.py 2s "${names[@]}"
"$stage_py" -u pipeline.py 2s "${names[@]}"
representative=$("$stage_py" freeze.py)
"$stage_py" -u pipeline.py extrapolated3 "$representative" gru lstm tcn transformer
"$stage_py" -u evaluate.py extrapolated3
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
mkdir -p "$stage_out/trained3"
exec bash start_three_queue.sh
