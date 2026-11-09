#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
branch=research/pacld-crm-training-comparison-v2
[[ "$(git branch --show-current)" == "$branch" ]]
# Explicit scope only. .gitignore excludes window identities, checkpoints and raw outputs.
git add scripts/pacld_crm_v2.py scripts/verify_pacld_crm_v2.py scripts/train_pacld_crm_v2.py scripts/evaluate_pacld_crm_v2.py scripts/report_pacld_crm_v2.py scripts/run_pacld_crm_v2.sh scripts/publish_pacld_crm_v2.sh experiments/phase_anchored_coupled_latent_v1/crm_training_comparison_v2_zh.md experiments/phase_anchored_coupled_latent_v1/crm_training_comparison_v2
# Abort if unrelated files were staged by another session.
while IFS= read -r path; do
  case "$path" in
    scripts/pacld_crm_v2.py|scripts/verify_pacld_crm_v2.py|scripts/train_pacld_crm_v2.py|scripts/evaluate_pacld_crm_v2.py|scripts/report_pacld_crm_v2.py|scripts/run_pacld_crm_v2.sh|scripts/publish_pacld_crm_v2.sh|experiments/phase_anchored_coupled_latent_v1/crm_training_comparison_v2_zh.md|experiments/phase_anchored_coupled_latent_v1/crm_training_comparison_v2/.gitignore|experiments/phase_anchored_coupled_latent_v1/crm_training_comparison_v2/training_contract.json|experiments/phase_anchored_coupled_latent_v1/crm_training_comparison_v2/preflight.json|experiments/phase_anchored_coupled_latent_v1/crm_training_comparison_v2/loss_calibration.json|experiments/phase_anchored_coupled_latent_v1/crm_training_comparison_v2/sampling_summary.json|experiments/phase_anchored_coupled_latent_v1/crm_training_comparison_v2/training_progress.csv|experiments/phase_anchored_coupled_latent_v1/crm_training_comparison_v2/training_comparison.csv|experiments/phase_anchored_coupled_latent_v1/crm_training_comparison_v2/moment_comparison.csv|experiments/phase_anchored_coupled_latent_v1/crm_training_comparison_v2/state_prediction_comparison.csv|experiments/phase_anchored_coupled_latent_v1/crm_training_comparison_v2/cir_comparison.csv|experiments/phase_anchored_coupled_latent_v1/crm_training_comparison_v2/a2_comparison.csv|experiments/phase_anchored_coupled_latent_v1/crm_training_comparison_v2/result_summary.json|experiments/phase_anchored_coupled_latent_v1/crm_training_comparison_v2/*.png) ;;
    *) echo "Refusing unrelated or non-public staged file: $path" >&2; exit 3 ;;
  esac
done < <(git diff --cached --name-only)
git diff --cached --check
git diff --cached --stat
git diff --cached --name-only
if ! git diff --cached --quiet; then git commit -m "Record fixed-budget PACLD CRM v2 comparison and calibration status"; fi
git push -u origin "$branch"
local_sha=$(git rev-parse HEAD)
remote_sha=$(git ls-remote --heads origin "$branch" | cut -f1)
[[ "$local_sha" == "$remote_sha" ]]
echo "REMOTE_VERIFIED $local_sha"
