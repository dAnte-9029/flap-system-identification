# September trajectory dataset v2

Dataset ID: `trajectory_v2_september_phase_observed`.
Authority: user-approved September-only Step 1, with explicit source inventory in
`configs/data/trajectory_september_v2.yaml`. This is a new trajectory dataset,
not a replacement or ratio-8 rebuild of the April canonical dataset. The April
canonical registry and August v1 remain unchanged. No aerodynamic prior is used.

## Frozen scope and split

- Train: six admitted September 6 logs; validation: nine September 7 logs.
- Sealed test: six admitted September 8 log paths, inventory only. No test ULog,
  test sample, or test statistic is read/computed by this builder.
- Three previously flagged short/incomplete logs remain excluded. Their reasons
  and the previous September 8 quality-audit exposure are recorded in metadata.
- Source firmware, structural parameters and output functions must match config.
  A missing file or contract mismatch fails; no directory discovery or fallback.
- Only train/validation materialization is supported at this stage. There is no
  test-unlock CLI flag. Future evaluation requires a separately reviewed change.

## Observations and alignment

Reference clock is **publication `timestamp`**, on native approximately 50 Hz
`vehicle_local_position` samples. Other signals use publication-time past-only
zero-order hold. `timestamp_sample` is retained as diagnostic metadata for the
reference state, not used to backdate when an observation became available.
Unlike August v1's event-time convention, this excludes packets published after
the prediction origin. Measurements remain delayed/noisy PX4 estimates, not
instantaneous independent ground truth. Publication-minus-sample lag is reported.

Rigid-body states, controls, freshness limits and safe-flight masks are reused
from v1: local NED position/velocity, body-FRD to NED quaternion, body-FRD rates,
four normalized commands in order motor/left elevon/right elevon/rudder.
No filter, normalization or parameter fit uses validation or test data.

Core rows require valid fresh state, all controls, relative encoder phase and
0.5–20 Hz flap frequency, armed airborne safe status. Break segments at invalid
rows, state/attitude resets, encoder reversal, mode change and gaps above 50 ms.
Mandatory topic timestamps must increase strictly; malformed streams fail.

## Phase and frequency

- Logged ratio is 7.909091; encoder counts per motor revolution are 4096. No
  April ratio-8 rescaling is applied. Consecutive Hall pulses versus interpolated
  encoder counts are checked descriptively; that offline diagnostic is never
  an input feature or a future-dependent row mask.
- The existing relative encoder phase is retained for the core baseline.
- `logged_flap_phase_rad` and sin/cos use causal, finite, flagged-valid
  `wing_phase.phase_rad` in [0, 2*pi). Invalid phase is NaN, never zero-filled.
- `valid_logged_phase` means a valid logged phase packet, not verified physical
  wing pose. `hall_reference_valid` additionally requires a preceding logged
  Hall event within 3 s **at the phase packet timestamp**. This flag cannot
  retroactively validate a phase packet using a later Hall event.
- Physical zero pose/direction and cross-flight installation consistency are
  currently unconfirmed. `absolute_flap_phase_*` remain invalid. A verified
  mechanical-zero claim requires user calibration information and a new frozen
  contract. No April physical-zero definition is assumed.
- `flap_frequency_hz` retains the v1 wing_phase/flap_frequency source policy;
  RPM0/1 are diagnostic only, not new mandatory inputs.

## Windows, history, and leakage boundary

Generate 1, 2, 3 and 5 s windows with nominal 50 Hz and 0.2 s stride, retaining
actual timestamp increments and observed horizons. Inclusive states: N+1;
exclusive-final controls: N. Every window preserves log/segment/sample keys.

Three explicit cohorts are produced: `core`, `logged_phase`, `hall_phase`.
Each phase cohort is resegmented at its own invalid rows before windowing. These
indices enable comparisons on identical windows without dropping the no-Hall
core logs. Consumers must name a cohort and horizon; no implicit phase fallback.
History can only be selected at/before t0 within that cohort's same segment.
Window metadata records available history duration for later 0.5/1 s eligibility.

Observable input: causal state/flap history and initial state. Optional causal
air-data context remains explicitly masked. Future-known input: only the four
commands from t0 through tT exclusive. Future actual state, phase, frequency,
air-data, Hall events, quality fields and cohort masks are forbidden inputs.
Quality masks select windows, never predict dynamics. Targets: rigid-body state
and optionally phase/frequency; future targets are only used in losses/metrics.

## Provenance and outputs

Manifest records resolved config/source paths, config and source-audit hashes,
processed train/validation ULog hashes, firmware/config/channel checks, time and
phase diagnostics, split inventory, sample/window hashes, code identity and
Python environment. No sealed-test content hash is computed.

Versioned output refuses to overwrite a nonempty directory. Registry entry pins
the produced manifest hash. Consumers must verify registered manifest and artifact
hashes, name train/validation explicitly, and preserve sealed-test isolation.
Full samples stay under ignored `dataset/`; compact build evidence is stored in
`docs/audits/results/`. This step trains no model and creates no Git commit.

## Build result and reproduction (2026-09-10)

Registry: `configs/data/trajectory_dataset_registry.yaml`; manifest:
`dataset/trajectory_v2_september_phase_observed/manifest.json`.

| Partition/cohort | Valid duration (s) | 1 s windows | 2 s | 3 s | 5 s |
| --- | ---: | ---: | ---: | ---: | ---: |
| train/core | 923.205 | 4,463 | 4,297 | 4,134 | 3,826 |
| train/logged phase | 777.187 | 3,750 | 3,604 | 3,459 | 3,181 |
| train/Hall-supported phase | 619.758 | 3,004 | 2,899 | 2,794 | 2,595 |
| validation/all three cohorts | 1,509.393 | 7,418 | 7,279 | 7,145 | 6,904 |

Source sample rows: 59,367 train and 98,208 validation; core valid rows: 46,196
and 75,502. Core train includes 6 logs / 35 segments; validation includes 9 logs /
32 segments. Logged-phase coverage of core samples is 84.19% train and 100%
validation. Hall-supported coverage is 67.13% train and 100% validation. The
stricter Hall cohort has four train logs; these are candidate comparison cohorts,
not three independent datasets whose window counts should be added as sample size.

All 13 processed logs with Hall passed the 1% median encoder/Hall ratio check.
Two train logs have no logged Hall. The logged-phase zero-to-Hall relationship was
previously checked in the source audit; its first-post-event angle includes sample
latency and does not calibrate physical wing pose. The numeric phase is available
for observed-phase modeling, but physical-zero and installation confirmation remain
open metadata requirements. No cross-flight absolute mechanical-pose claim is made.

The independent output audit checked all 128,244 window records across horizons
and cohorts: hashes, keys, entire source slices, core/phase masks, time gaps and
final-control exclusion. Publication-time phase and Hall joins were checked on
all samples. This count includes overlapping windows/cohorts, not independent
observations. Processed ULog identities equal the 15 train/validation log paths;
sealed-test samples and content hashes are absent.

Build (refuses to overwrite the existing output):

```bash
/home/zn/anaconda3/envs/flap-train-gpu/bin/python scripts/build_september_trajectory_dataset.py
```

Use `--output-root /tmp/september-reproduction` for a separate reproduction.
The registry remains pinned to the original build; rebuilding does not silently
change the registered identity. Verification helper:
`system_identification.data.september_trajectory.verify_registered_dataset`.

Tests: `tests/test_september_trajectory.py`, `tests/test_trajectory_dataset.py`,
and existing trajectory baseline/Main V1/Main V2 regression tests. Detailed counts:
`docs/audits/results/2026-09-10_september_dataset_build.json`; structural checks:
`docs/audits/results/2026-09-10_september_dataset_checks.json`.

V2 is intentionally not a drop-in input to the old v1 runners: consumers must
resolve this registry, explicitly select horizon/cohort, and join window endpoints
using `(log_id, sample_in_log)`. Cohort segment IDs must not be joined to the core
sample table's segment IDs. Matched ablations use the same selected window table
and history eligibility, including when the baseline does not consume phase.
