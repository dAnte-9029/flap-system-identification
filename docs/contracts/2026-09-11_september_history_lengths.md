# Matched causal history-length experiment

User authorized 0.5/1/2-second history comparison without monitoring. Use 26/51/101
samples at nominal 50 Hz, all ending at the same t0. Keep the registered September
train/validation split; September 8 remains sealed. Filter both partitions to
available_history_s >= 2 and start_sample_in_segment >= 100; require complete,
contiguous unpadded histories and preserve every flight log. Save common windows
and before/after coverage. All three models see exactly the same future targets
and control tapes; only past-history suffix length changes.

Use controlled GRU hidden size 64, 2-second objective, seed 17, 80 epochs, batch
256, AdamW lr 3e-4, weight decay 1e-5, gradient clip 5, GPU 1. Freeze 80 epochs
based on the preceding training-duration diagnostic; no validation-based early
stopping in this run. Equal initialization and shuffle seeds, same update count;
encoder computation differs with history length. Fit one common normalization
on the 101-sample training batch and training transitions, shared by all models.
Phase remains relative to t0; no future state, measured phase or frequency input.

Evaluate all matched training windows at 1/2 seconds and all matched validation
windows continuously at 1/2/3/5 seconds without resets. Save full error arrays,
per-window endpoint metrics, per-log and equal-log macro metrics, failure flags
and train-only speed/body-z-rate tertile groups with counts. Body z rate is a
turn-intensity proxy. Compare within this run: old 0.5-second results use different
windows and normalization. One seed cannot establish final scientific superiority.

Output: artifacts/september_history_lengths_20260911; sibling .run.log and
.launch.json. No overwrite, automatic retry, test access or interactive monitoring.
