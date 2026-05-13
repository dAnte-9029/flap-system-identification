# DeLaurier Residual Frequency Analysis Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Decompose the calibrated DeLaurier residual into frequency bands and measure which residual energy bands are reduced by the residual Transformer.

**Architecture:** Add a standalone analysis script that reads aligned residual predictions, forms `true_residual = label - prior` and `remaining_residual = true_residual - pred`, computes one-sided FFT energy per `log_id/segment_id`, and aggregates energy by target and frequency component. The analysis complements phase and flight-condition diagnostics by distinguishing low-frequency bias, wingbeat-periodic content, harmonics, and broadband high-frequency residuals.

**Tech Stack:** Python 3.11, pandas, NumPy FFT, matplotlib, pytest.

---

### Task 1: Frequency Energy Functions

**Files:**
- Create: `scripts/analyze_delaurier_residual_frequency.py`
- Create: `tests/test_analyze_delaurier_residual_frequency.py`

**Steps:**
1. Write a failing test with a synthetic 100 Hz signal containing low, flap-main, and high-frequency components.
2. Verify the energy table assigns most known energy to the expected bands.
3. Verify `remaining_energy_fraction_of_true` and `energy_reduction_fraction` are computed from true versus remaining residual.
4. Implement minimal FFT energy and grouped aggregation.
5. Run the targeted test.

### Task 2: CLI And Figures

**Files:**
- Modify: `scripts/analyze_delaurier_residual_frequency.py`

**Steps:**
1. Add CLI arguments: `--aligned-parquet`, `--output-dir`, `--flap-half-width-hz`, `--high-band-low-hz`, `--high-band-high-hz`.
2. Default frequency components:
   - `low_0_1hz`
   - `mid_1_3hz`
   - `flap_main`
   - `harmonic_2f`
   - `harmonic_3f`
   - `broadband_high_8_25hz_excl_structured`
3. Write:
   - `frequency_residual_energy.csv`
   - `frequency_residual_summary.csv`
   - `frequency_residual_config.json`
4. Plot grouped bars for key targets (`fx_b`, `fz_b`, `my_b`): true residual energy fraction and remaining residual energy fraction.

### Task 3: Run On Residual Transformer

**Command:**

```bash
conda run -n flap-train-gpu python scripts/analyze_delaurier_residual_frequency.py \
  --aligned-parquet artifacts/20260513_delaurier_residual_nn_v1/residual_transformer_h128_gpu_combined_eval/test_aligned_residual_predictions.parquet \
  --output-dir artifacts/20260513_delaurier_residual_nn_v1/residual_transformer_h128_gpu_frequency_analysis
```

Expected outputs:
- `frequency_residual_energy.csv`
- `frequency_residual_summary.csv`
- `frequency_residual_energy_key_targets.png`
- `frequency_residual_energy_key_targets.pdf`

### Task 4: Result Note

**Files:**
- Modify: `docs/results/2026-05-13-delaurier-residual-nn-gpu.md`

**Steps:**
1. Add a short frequency-domain residual section.
2. Report which bands dominate `fx_b` and `fz_b` DeLaurier residual energy.
3. Report how much energy remains after residual Transformer correction.
4. Bound the interpretation: broadband high-frequency content can include both physical structure and label/reconstruction noise.

### Task 5: Verification

Run:

```bash
conda run -n flap-train-gpu python -m pytest tests/test_analyze_delaurier_residual_frequency.py tests/test_analyze_delaurier_residual_conditions.py tests/test_analyze_delaurier_residual_phase.py -q
conda run -n flap-train-gpu python -m py_compile scripts/analyze_delaurier_residual_frequency.py
```

Inspect the generated CSV summaries before reporting conclusions.
