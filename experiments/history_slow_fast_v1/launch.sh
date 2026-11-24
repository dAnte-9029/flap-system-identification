#!/usr/bin/env bash
# Run both paired members on separate GPUs, preserving independent exit status and logs.
set -eu
src=$(cd "$(dirname "$0")" && pwd)
cache=$(realpath "$1")
run=$(realpath -m "$2")
mkdir "$run"
mkdir "$run/source"
cp "$src"/{prepare.py,model.py,train.py,launch.sh} "$run/source"
member() {
    local name=$1 gpu=$2 rc=0
    /home/zn/anaconda3/envs/flap-train-gpu/bin/python -u "$src/train.py" --cache "$cache" --out "$run/$name" --model "$name" --gpu "$gpu" > "$run/$name.log" 2>&1 || rc=$?
    echo "$rc" > "$run/$name.exit"
    return "$rc"
}
member slowfast 0 &
first=$!
member single 1 &
second=$!
rc=0
wait "$first" || rc=1
wait "$second" || rc=1
exit "$rc"
