# Frozen held-out evaluation registration

Frozen UTC: 2026-09-22T06:09:10.808392+00:00

Sep8 is the primary independent test; Sep19 is the supplementary held-out date. Both must be executed and reported separately irrespective of errors. This is not a claim of never-read raw logs: Sep8 previously received a descriptive quality audit. Prior access metadata is retained in protocol.json.

The exact15 checkpoints, seeds17/23/42, native-time quality gates, H26 shared origins, Actual/Hold conditions, original train control thresholds and statistical rules are registered in protocol.json. H26 Actual is inferred once per seed/date and reused across comparisons. No training, fitting, seed selection, ensemble, or validation/test-dependent filtering. Each date: within-flight vector/geodesic RMS, equal-flight mean, three-seed sample SD. B0 has no seed SD. Primary25steps, auxiliary5/10/50. H13 remains an ablation, H26 remains main; original validation Mixed classification unchanged.

Admission follows the original data-family gates: Sep8 inherits v2 Sep7 gate, Sep19 inherits v3 Sep17 gate; common native stride50 windows require25 prior samples and50future transitions. See exact source hashes and rule details in protocol.json. No test-sequence read occurs before this file and protocol are frozen.

Frozen train thresholds: [{"horizon_s": 0.1, "step": 5, "train_q25": 0.0695041439687429, "train_q75": 0.1259374744559026, "grouping_available": true, "n_train_origins": 28293}, {"horizon_s": 0.2, "step": 10, "train_q25": 0.1100887242608976, "train_q75": 0.1936055708369762, "grouping_available": true, "n_train_origins": 28293}, {"horizon_s": 0.5, "step": 25, "train_q25": 0.1735699983872748, "train_q75": 0.3216767771852724, "grouping_available": true, "n_train_origins": 28293}, {"horizon_s": 1.0, "step": 50, "train_q25": 0.2541643074147546, "train_q75": 0.5027411687346327, "grouping_available": true, "n_train_origins": 28293}]

No qualitative case is generated; all quantitative metrics/dates are shown. Finite but poor predictions are retained. Numerical/runtime failures remain explicit missing full results, not filtered samples. Engineering fixes require append-only amendment with results exposure; no scientific threshold or tuning. These tests cannot establish arbitrary-action causality, closed-loop benefits, or long-horizon stability.
