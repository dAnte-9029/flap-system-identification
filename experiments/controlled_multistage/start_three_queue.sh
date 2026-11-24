#!/usr/bin/env bash
set -euo pipefail
stage_src=$(cd "$(dirname "$0")" && pwd)
cd "$stage_src"
stage_out=/home/zn/flap-system-identification/outputs/controlled_multistage_20261120
stage_py=/home/zn/anaconda3/envs/flap-train-gpu/bin/python
representative=$("$stage_py" -c 'from workspace import RUN; import json; print(json.loads((RUN/"freeze.json").read_text())["representative"])')
train_one() {
    local member="$1" gpu="$2"
    "$stage_py" -u train_three.py "$member" --gpu "$gpu" > "$stage_out/trained3/$member.log" 2>&1
}
( train_one transformer 1 && train_one "$representative" 1 ) &
attention_queue=$!
rc=0
( train_one tcn 0 && train_one gru 0 && train_one lstm 0 ) || rc=1
wait "$attention_queue" || rc=1
if (( rc )); then exit "$rc"; fi
"$stage_py" -u evaluate.py trained3
"$stage_py" -u pipeline.py trained3 "$representative" gru lstm tcn transformer
"$stage_py" figures.py
