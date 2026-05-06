# SUBNET Three-Stage Rollout Comparison

## Dataset and Split

```text
dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1
split_policy: whole_log
```

## Compared Recipes

```text
mlp_paper_no_accel_v2
causal_gru_paper_no_accel_v2_phase_actuator_airdata
subsection_gru_paper_no_accel_v2_phase_actuator_airdata
subnet_discrete_paper_no_accel_v2_phase_actuator_airdata
ct_subnet_euler_paper_no_accel_v2_phase_actuator_airdata
```

The rollout models use:

```text
sequence_history_size: 64
rollout_size: 32
rollout_stride: 32
sequence_feature_mode: phase_actuator_airdata
current_feature_mode: remaining_current
latent_size: 16
ct_subnet_euler dt_over_tau: 0.01
```

All rollout configs record:

```text
has_acceleration_inputs: false
has_velocity_history: false
has_angular_velocity_history: false
has_alpha_beta_history: false
uses_whole_log_split: true
```

## Commands

Tau sweep:

```bash
for r in 0.01 0.03 0.1 0.3; do
  python scripts/run_baseline_comparison.py \
    --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
    --output-dir artifacts/20260506_ct_subnet_tau_${r} \
    --recipes ct_subnet_euler_paper_no_accel_v2_phase_actuator_airdata \
    --max-train-samples 65536 \
    --max-val-samples 32768 \
    --max-test-samples 32768 \
    --sequence-history-size 64 \
    --rollout-size 32 \
    --rollout-stride 32 \
    --latent-size 16 \
    --dt-over-tau "$r" \
    --hidden-sizes 128,128 \
    --batch-size 512 \
    --max-epochs 12 \
    --early-stopping-patience 4 \
    --learning-rate 0.001 \
    --weight-decay 0.00001 \
    --device cuda:0
done
```

Final comparison:

```bash
python scripts/run_baseline_comparison.py \
  --split-root dataset/canonical_v0.2_training_ready_split_hq_v3_direct_airspeed_logsplit_paper_v1 \
  --output-dir artifacts/20260506_subnet_three_stage_final \
  --recipes mlp_paper_no_accel_v2 causal_gru_paper_no_accel_v2_phase_actuator_airdata subsection_gru_paper_no_accel_v2_phase_actuator_airdata subnet_discrete_paper_no_accel_v2_phase_actuator_airdata ct_subnet_euler_paper_no_accel_v2_phase_actuator_airdata \
  --batch-size 512 \
  --max-epochs 50 \
  --early-stopping-patience 8 \
  --learning-rate 0.001 \
  --weight-decay 0.00001 \
  --hidden-sizes 128,128 \
  --sequence-history-size 64 \
  --rollout-size 32 \
  --rollout-stride 32 \
  --latent-size 16 \
  --dt-over-tau 0.01 \
  --device cuda:0
```

## Tau Sweep

```text
dt_over_tau  best_epoch  val_rmse  test_rmse  test_r2
0.01                 12  1.734373   1.710510  0.433096
0.03                 12  1.744751   1.718366  0.427754
0.10                 12  1.745698   1.726084  0.419652
0.30                 12  1.759484   1.745690  0.414228
```

`dt_over_tau=0.01` was selected for the final CT run.

## Overall Metrics

```text
recipe                                                model_type       best_epoch  test_mae  test_rmse  test_r2
mlp_paper_no_accel_v2                                mlp                    20    0.574047   1.198693  0.652377
causal_gru_paper_no_accel_v2_phase_actuator_airdata  causal_gru             15    0.482974   1.007040  0.709279
subsection_gru_paper_no_accel_v2_phase_actuator_airdata subsection_gru       45    0.527352   1.125607  0.670223
subnet_discrete_paper_no_accel_v2_phase_actuator_airdata subnet_discrete     50    0.547245   1.148124  0.668950
ct_subnet_euler_paper_no_accel_v2_phase_actuator_airdata ct_subnet_euler     36    0.601968   1.257864  0.636187
```

## Per-Target R2

```text
recipe                                                fx_b    fy_b    fz_b    mx_b    my_b    mz_b
mlp_paper_no_accel_v2                                0.9190  0.2476  0.9426  0.4768  0.8437  0.4847
causal_gru_paper_no_accel_v2_phase_actuator_airdata  0.9388  0.3455  0.9634  0.5583  0.8943  0.5553
subsection_gru_paper_no_accel_v2_phase_actuator_airdata 0.9177 0.2701 0.9536 0.5091 0.8610 0.5099
subnet_discrete_paper_no_accel_v2_phase_actuator_airdata 0.9164 0.2760 0.9503 0.5056 0.8480 0.5174
ct_subnet_euler_paper_no_accel_v2_phase_actuator_airdata 0.9129 0.2343 0.9340 0.4466 0.8248 0.4645
```

## Per-Target RMSE

```text
recipe                                                fx_b      fy_b      fz_b      mx_b      my_b      mz_b
mlp_paper_no_accel_v2                                1.420171  1.283642  2.226332  0.002407  0.001710  0.000369
causal_gru_paper_no_accel_v2_phase_actuator_airdata  1.226997  1.189067  1.779149  0.002213  0.001406  0.000342
subsection_gru_paper_no_accel_v2_phase_actuator_airdata 1.419886 1.255834 2.002183 0.002335 0.001613 0.000360
subnet_discrete_paper_no_accel_v2_phase_actuator_airdata 1.430725 1.250749 2.073108 0.002343 0.001687 0.000357
ct_subnet_euler_paper_no_accel_v2_phase_actuator_airdata 1.460526 1.286274 2.388656 0.002479 0.001811 0.000376
```

## Worst Per-Log Cases

```text
recipe                                                worst_log                       test_r2  test_rmse
mlp_paper_no_accel_v2                                log_4_2026-4-12-17-43-30        0.513706 1.251973
causal_gru_paper_no_accel_v2_phase_actuator_airdata  log_4_2026-4-12-17-43-30        0.545949 1.237152
subsection_gru_paper_no_accel_v2_phase_actuator_airdata log_4_2026-4-12-17-43-30     0.489320 1.255895
subnet_discrete_paper_no_accel_v2_phase_actuator_airdata log_4_2026-4-12-17-43-30    0.495187 1.237910
ct_subnet_euler_paper_no_accel_v2_phase_actuator_airdata log_4_2026-4-12-17-43-30    0.503192 1.302793
```

## Worst Regime Bins

Regime diagnostics for sequence and rollout models were computed from full-split predictions, then binned by aligned prediction metadata.

```text
recipe                                                worst_regime                         bin             test_r2  test_rmse
mlp_paper_no_accel_v2                                cycle_flap_frequency_hz              (-0.001, 3.0]   0.079440 1.353678
causal_gru_paper_no_accel_v2_phase_actuator_airdata  cycle_flap_frequency_hz              (-0.001, 3.0]   0.508076 0.729493
subsection_gru_paper_no_accel_v2_phase_actuator_airdata cycle_flap_frequency_hz           (-0.001, 3.0]   0.462855 0.848401
subnet_discrete_paper_no_accel_v2_phase_actuator_airdata cycle_flap_frequency_hz          (-0.001, 3.0]   0.393163 0.821487
ct_subnet_euler_paper_no_accel_v2_phase_actuator_airdata phase_corrected_rad              (1.571, 3.142]  0.461534 1.229239
```

## Stage Decisions

```text
Stage 1 subsection_gru:
  Beats MLP overall.
  Does not beat causal GRU overall or per-target.
  Keep as an ablation, not the default.

Stage 2 subnet_discrete:
  Beats MLP overall.
  Does not beat causal GRU.
  Slightly worse than subsection_gru overall.
  Keep as a research branch only.

Stage 3 ct_subnet_euler:
  Does not beat MLP.
  Does not beat causal GRU.
  Current Euler CT formulation is not a good default.
```

## Recommendation

The default forward sequence baseline should remain:

```text
causal_gru_paper_no_accel_v2_phase_actuator_airdata
```

The SUBNET-inspired models are implemented and diagnostically covered, but this run does not justify replacing causal GRU. The most useful follow-up is a smaller ablation around Stage 1 only: rollout size 16 vs 32, hidden size 128 vs 256, and maybe a loss that weights near-term and far-term rollout steps separately.

## Artifacts

```text
artifacts/20260506_subnet_three_stage_final/baseline_comparison_summary.csv
artifacts/20260506_subnet_three_stage_final/baseline_comparison_summary.json
artifacts/20260506_subnet_three_stage_final/baseline_comparison_summary.png
artifacts/20260506_subnet_three_stage_final/<recipe>/diagnostics/per_log_metrics.csv
artifacts/20260506_subnet_three_stage_final/<recipe>/diagnostics/per_regime_metrics.csv
```

## Caveats

The rollout models use non-overlapping test subsections, so their `test_sample_count` differs slightly from MLP and causal GRU. This is intentional to avoid duplicated rollout targets in headline metrics.

The CT model only implements Euler integration. RK4 may be worth trying later, but only after Stage 1 ablations clarify whether multi-step rollout loss itself is helpful.

