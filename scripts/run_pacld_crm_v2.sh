#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:scripts
export MPLCONFIGDIR=/tmp/pacld-crm-v2-matplotlib
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
[[ "$CUDA_VISIBLE_DEVICES" != *,* ]] || { echo 'Exactly one GPU is allowed'; exit 2; }
CRM_PY=/home/zn/anaconda3/envs/flap-train-gpu/bin/python
out=experiments/phase_anchored_coupled_latent_v1/crm_training_comparison_v2
mkdir -p "$out/local"
exec 9>"$out/local/pipeline.lock"
flock -n 9 || { echo 'CRM-v2 pipeline already running'; exit 2; }
echo "$$" > "$out/local/pipeline.pid"
resume=()
if [[ ${1:-} == --resume ]]; then resume=(--resume); elif [[ $# != 0 ]]; then echo 'Only --resume is supported'; exit 2; fi
trap 'rc=$?; if [[ $rc != 0 ]] && ! grep -q '^blocked' "$out/local/pipeline_status.txt"; then printf "failed exit=%s time=%s\n" "$rc" "$(date -Is)" > "$out/local/pipeline_status.txt"; fi' EXIT
"$CRM_PY" -c 'from pacld_crm_v2 import initialize; initialize()'
printf 'preflight %s\n' "$(date -Is)" > "$out/local/pipeline_status.txt"
if [[ ! -f "$out/loss_calibration.json" ]]; then
  if ! "$CRM_PY" scripts/verify_pacld_crm_v2.py; then
    "$CRM_PY" scripts/report_pacld_crm_v2.py
    if [[ -f "$out/loss_calibration.json" ]]; then bash scripts/publish_pacld_crm_v2.sh; echo blocked_by_calibration > "$out/local/pipeline_status.txt"; fi
    exit 2
  fi
fi
if ! "$CRM_PY" -c 'import json,sys; from pacld_crm_v2 import OUT; ok=all(json.loads((OUT/p).read_text())["status"]=="PASS" for p in ["preflight.json","loss_calibration.json"]); print("training gates PASS" if ok else "BLOCKED: preflight/calibration forbids formal training"); sys.exit(0 if ok else 2)'; then
  "$CRM_PY" scripts/report_pacld_crm_v2.py
  echo 'blocked_by_calibration' > "$out/local/pipeline_status.txt"
  exit 2
fi
for group in B0 B1 B2; do
  printf 'training %s %s\n' "$group" "$(date -Is)" > "$out/local/pipeline_status.txt"
  "$CRM_PY" scripts/train_pacld_crm_v2.py --group "$group" "${resume[@]}"
done
printf 'evaluating %s\n' "$(date -Is)" > "$out/local/pipeline_status.txt"
"$CRM_PY" scripts/evaluate_pacld_crm_v2.py
"$CRM_PY" scripts/report_pacld_crm_v2.py
printf 'publishing %s\n' "$(date -Is)" > "$out/local/pipeline_status.txt"
bash scripts/publish_pacld_crm_v2.sh
printf 'complete %s\n' "$(date -Is)" > "$out/local/pipeline_status.txt"
