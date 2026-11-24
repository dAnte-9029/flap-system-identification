#!/usr/bin/env bash
set -eu
src=$(cd "$(dirname "$0")" && pwd)
cache=$(realpath "$1")
tapes=$(realpath "$2")
out=$(realpath -m "$3")
mkdir "$out"
mkdir -p "$out/source/controlled_slow_fast_v1" "$out/source/history_slow_fast_v1"
cp "$src"/{controls.py,dynamics.py,backbones.py,fit.py,compare.py,compare_suite.py,launch_baselines.sh,test_controlled.py} "$out/source/controlled_slow_fast_v1/"
cp "$src"/../history_slow_fast_v1/{train.py,model.py} "$out/source/history_slow_fast_v1/"
member() {
    local model=$1 gpu=$2 rc=0
    /home/zn/anaconda3/envs/flap-train-gpu/bin/python -u "$src/fit.py" --model "$model" --gpu "$gpu" --cache "$cache" --tapes "$tapes" --out "$out/$model" > "$out/$model.log" 2>&1 || rc=$?
    echo "$rc" > "$out/$model.exit"
    return "$rc"
}
(member lstm 0 && member tcn 0) &
queue=$!
member transformer 1 &
attention=$!
rc=0
wait "$queue" || rc=1
wait "$attention" || rc=1
exit "$rc"
