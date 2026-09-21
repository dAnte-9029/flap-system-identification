"""Paper Step 1: auditable train/open-validation-only baseline pilot.

Run audit first, then run with an explicitly chosen training schedule. Existing
Ours weights and data are read-only. No globbing of datasets or test artifacts.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import torch
import yaml

from system_identification.data.september_trajectory import file_hash
from system_identification.evaluation.paper_baselines import (
    HORIZONS, aggregate_flights, cohort, endpoint_metrics,
)
from system_identification.models.paper_baselines import MemorylessTrajectoryModel
from system_identification.models.trajectory import ConstantTwistPredictor
from system_identification.models.trajectory_main_v1 import CausalHistoryTrajectoryModel
from system_identification.models.trajectory_main_v2 import ActuatorAwareTrajectoryModel
from system_identification.training.main_v2_increment import train_increment_stage
from system_identification.training.trajectory_main_v1 import (
    _model_call, assemble_history_trajectory_windows, fit_main_v1_stats,
    predict_history_trajectory_model,
)

OUT = ROOT / "docs/analysis/results/paper_baseline_comparison_v1"
ART = ROOT / "artifacts/paper_baseline_comparison_v1"
OURS = ROOT / "artifacts/september_expanded_main_v2/model.pt"
OURS_SHA = "750b53b7b5321b884a929dd6abc3fb4cbebf116eafbaa13c4a1add16e39658eb"
DATASET_ID = "trajectory_v3_september_expanded"
STATS = ("feature_mean", "feature_std", "control_mean", "control_std", "derivative_mean", "derivative_std")
MODELS = ("B0_ConstantVelocity", "B1_MLP", "B2_StandardGRU", "B3_ActuatorAwareGRU")
SOURCES = [
    "scripts/run_paper_baseline_comparison.py",
    "scripts/report_paper_baseline_comparison.py",
    "src/system_identification/models/paper_baselines.py",
    "src/system_identification/evaluation/paper_baselines.py",
    "src/system_identification/models/neural.py",
    "src/system_identification/models/trajectory.py",
    "src/system_identification/models/trajectory_main_v1.py",
    "src/system_identification/models/trajectory_main_v2.py",
    "src/system_identification/models/main_v2_simulator.py",
    "src/system_identification/training/main_v2_increment.py",
    "src/system_identification/training/trajectory_main_v1.py",
    "src/system_identification/training/trajectory_baselines.py",
    "src/system_identification/evaluation/trajectory.py",
    "scripts/train_expanded_september.py",
]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def status(stage, **values):
    row = dict(stage=stage, pid=os.getpid(), unix_time=time.time(), **values)
    status_name = os.environ.get('PAPER_BASELINE_STATUS_NAME', 'status')
    write_json(ART / f"{status_name}.json", row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def load_ours():
    if file_hash(OURS) != OURS_SHA:
        raise ValueError("frozen Ours checkpoint hash mismatch")
    checkpoint = torch.load(OURS, map_location="cpu", weights_only=False)
    stats = {k: checkpoint["state_dict"]["base_model." + k].numpy() for k in STATS}
    base = CausalHistoryTrajectoryModel(**checkpoint["base_config"], **stats)
    model = ActuatorAwareTrajectoryModel(
        base_model=base, tail_mean=checkpoint["tail_mean"], tail_std=checkpoint["tail_std"],
        **checkpoint["config"],
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval(), stats, checkpoint["protocol"]


def build_model(name, stats):
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    if name == "B1_MLP":
        return MemorylessTrajectoryModel(**stats)
    if name == "B2_StandardGRU":
        return CausalHistoryTrajectoryModel(hidden_size=64, use_controls=True, **stats)
    raise ValueError(name)


def parameter_counts(model):
    return dict(total=sum(p.numel() for p in model.parameters()),
                trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))


def verify_sources():
    return {p: file_hash(ROOT / p) for p in SOURCES if (ROOT / p).exists()}


def prepare():
    """Verify explicit manifest/partition allowlist BEFORE opening sample files."""
    reg_path = ROOT / "configs/data/trajectory_dataset_registry.yaml"
    registry = yaml.safe_load(reg_path.read_text())
    if registry["default_dataset_id"] != DATASET_ID:
        raise ValueError("registry changed: this frozen pilot must be reviewed")
    entry = registry["datasets"][DATASET_ID]
    manifest_path = ROOT / entry["manifest_path"]
    if file_hash(manifest_path) != entry["manifest_sha256"]:
        raise ValueError("registered manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    split = manifest["split_contract"]["assignments"]
    if manifest["split_contract"]["sealed_test_opened"]:
        raise ValueError("dataset declares opened test")
    train_ids, val_ids = set(split["train"]), set(split["validation"])
    excluded = set(split["sealed_test"]) | set(split["reserved_evaluation"])
    if train_ids & val_ids or (train_ids | val_ids) & excluded:
        raise ValueError("flight split overlap")
    source_hashes = manifest["source"]["ulog_sha256"]
    if len({source_hashes[k] for k in train_ids | val_ids}) != len(train_ids | val_ids):
        raise ValueError("duplicate source flight bytes")
    for log in val_ids:
        cohort(log)
    allowed = {f"{kind}_{part}.parquet" for kind in ("samples", "windows") for part in ("train", "validation")}
    if set(manifest["artifact_sha256"]) != allowed:
        raise ValueError("unexpected artifact manifest; no additional partition will be opened")
    for name in sorted(allowed):
        if file_hash(manifest_path.parent / name) != manifest["artifact_sha256"][name]:
            raise ValueError(f"artifact hash mismatch: {name}")
    ours, stats, original = load_ours()
    if original["manifest_sha256"] != file_hash(manifest_path) or original["sample_artifact_sha256"] != manifest["artifact_sha256"]:
        raise ValueError("Ours training provenance does not match benchmark dataset")
    identity = dict(manifest_sha256=file_hash(manifest_path), checkpoint_sha256=OURS_SHA,
                    loader_hashes={p: file_hash(ROOT / p) for p in [
                        "src/system_identification/training/trajectory_main_v1.py",
                        "src/system_identification/evaluation/trajectory.py",
                        "src/system_identification/training/trajectory_baselines.py"]})
    cache = ART / "prepared.pt"
    if cache.exists():
        prepared = torch.load(cache, map_location="cpu", weights_only=False)
        if prepared["identity"] != identity:
            raise ValueError("prepared data cache no longer matches frozen inputs")
        return prepared, ours, stats, original
    batches, coverage, origin_frames, checks = {}, [], {}, {}
    for part in ("train", "validation"):
        status("assembling", partition=part)
        samples = pd.read_parquet(manifest_path.parent / f"samples_{part}.parquet")
        windows = pd.read_parquet(manifest_path.parent / f"windows_{part}.parquet")
        if set(samples.log_id) != set(split[part]) or set(windows.log_id) != set(split[part]):
            raise ValueError(f"partition flight list mismatch: {part}")
        if set(samples.split) != {part}:
            raise ValueError("sample partition labels disagree")
        if windows.window_id.duplicated().any() or set(windows.state_sample_count) != {51}:
            raise ValueError("invalid window identities/lengths")
        batch = assemble_history_trajectory_windows(samples, windows, history_steps=26)
        if not batch.history_mask.all():
            raise ValueError("incomplete history")
        for values in vars(batch.trajectory.truth).values():
            if not np.isfinite(values).all():
                raise ValueError("nonfinite truth")
        for values in [batch.history_state_features, batch.history_controls, batch.trajectory.controls]:
            if not np.isfinite(values).all():
                raise ValueError("nonfinite model inputs")
        valid = samples.loc[samples.valid_core]
        keys = ["log_id", "segment_id", "sample_in_segment"]
        lookup = valid.set_index(keys)
        if lookup.index.duplicated().any():
            raise ValueError("duplicate sample identity")
        def at(offset):
            index = pd.MultiIndex.from_arrays([windows.log_id, windows.segment_id,
                                              windows.start_sample_in_segment + offset], names=keys)
            return lookup.loc[index, "timestamp_us"].to_numpy()
        origins = windows.copy()
        origins["history_start_timestamp_us"] = at(-25)
        origins["origin_timestamp_us"] = at(0)
        origins["end_timestamp_us"] = at(50)
        origins['history_span_s'] = (origins.origin_timestamp_us - origins.history_start_timestamp_us)*1e-6
        if not np.allclose((origins.end_timestamp_us - origins.origin_timestamp_us)*1e-6,
                           batch.trajectory.dt_s.sum(axis=1), rtol=0, atol=1e-12):
            raise ValueError("target timestamp misalignment")
        timing = []
        for nominal, step in HORIZONS.items():
            actual = (at(step)-at(0))*1e-6
            if not np.allclose(actual, batch.trajectory.dt_s[:, :step].sum(axis=1), rtol=0, atol=1e-12):
                raise ValueError("endpoint timestamp misalignment")
            timing.append(dict(partition=part, nominal_horizon_s=nominal, steps=step,
                               actual_min_s=float(actual.min()), actual_median_s=float(np.median(actual)),
                               actual_max_s=float(actual.max()), actual_mean_s=float(actual.mean())))
        pd.DataFrame(timing).to_csv(OUT / f'{part}_horizon_timing.csv',index=False)
        source_cols = [c for c in samples if c.endswith("source_timestamp_us")]
        availability = {}
        for c in source_cols:
            observed = valid[c].to_numpy()
            mask = np.isfinite(observed) & (observed >= 0)
            future_count = int(np.sum(observed[mask] > valid.timestamp_us.to_numpy()[mask]))
            availability[c] = future_count
            if future_count:
                raise ValueError(f"future source timestamp: {part}/{c}")
        checks[part] = dict(windows=len(windows), flights=len(split[part]),
                            full_history=True, finite=True, nominal_dt_s=.02,
                            dt_min_s=float(batch.trajectory.dt_s.min()), dt_max_s=float(batch.trajectory.dt_s.max()),
                            history_span_min_s=float(origins.history_span_s.min()), history_span_max_s=float(origins.history_span_s.max()),
                            endpoint_timestamps_match_integrated_dt=True,
                            max_truth_quaternion_norm_error=float(np.max(np.abs(np.linalg.norm(batch.trajectory.truth.quaternion_nb, axis=-1)-1))),
                            future_source_timestamp_counts=availability)
        if part == "train":
            fresh = fit_main_v1_stats(samples, batch)
            for k in STATS:
                if not np.array_equal(getattr(fresh, k).astype(np.float32), stats[k]):
                    raise ValueError(f"train-only normalization does not reproduce Ours: {k}")
            scales = [float(np.sqrt(np.mean(np.sum((getattr(batch.trajectory.truth, k)[:, 2] - getattr(batch.trajectory.truth, k)[:, 0])**2, axis=-1)))) for k in ("velocity_n", "angular_velocity_b")]
            if not np.allclose(scales, original["scales"], rtol=1e-12, atol=0):
                raise ValueError("increment scales differ from frozen training")
        else:
            origins["cohort"] = origins.log_id.map(cohort)
        origins.to_csv(OUT / f"{part}_origins.csv", index=False)
        for log, g in windows.groupby("log_id"):
            coverage.append(dict(partition=part, log_id=log, n_windows=len(g),
                                 cohort=cohort(log) if part == "validation" else "train"))
        batches[part], origin_frames[part] = batch, origins
    prepared = dict(identity=identity, batches=batches, origins=origin_frames, coverage=coverage,
                    checks=checks, manifest=manifest, registry_sha256=file_hash(reg_path),
                    manifest_path=str(manifest_path.relative_to(ROOT)))
    torch.save(prepared, cache)
    return prepared, ours, stats, original


def audit(prepared, ours, stats, original):
    counts = {MODELS[0]: dict(total=0, trainable=0)}
    for name in MODELS[1:3]:
        counts[name] = parameter_counts(build_model(name, stats))
    counts[MODELS[3]] = parameter_counts(ours)
    counts[MODELS[3]]["base_stage_trainable"] = sum(p.numel() for p in ours.base_model.parameters())
    manifest = prepared["manifest"]
    protocol = dict(
        experiment="paper_baseline_comparison_v1", status="audited_schedule_pending",
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        dataset_id=DATASET_ID, manifest_path=prepared["manifest_path"],
        manifest_sha256=prepared["identity"]["manifest_sha256"],
        artifact_sha256=manifest["artifact_sha256"], registry_sha256=prepared["registry_sha256"],
        train_flights=manifest["split_contract"]["assignments"]["train"],
        validation_flights=manifest["split_contract"]["assignments"]["validation"],
        excluded_sealed_flights={k: v for k, v in manifest["split_contract"]["assignments"].items() if k not in ("train", "validation")},
        admission_exclusions=[r for r in manifest["admission"] if r["status"] != "admitted"],
        phase_contract=manifest["phase_contract"], frequency_contract=manifest["frequency_contract"],
        model_phase="inherited relative_flap_phase_rad, relative to origin; not absolute logged Hall pose",
        frames=manifest["frames"], parameter_counts=counts,
        models={
            MODELS[0]: "Existing ConstantTwistPredictor: constant NED velocity and FRD body rate; exact quaternion exponential",
            MODELS[1]: "Existing MLPRegressor 16->64 ReLU->64 ReLU->7, current state+control; Main V1 integration; no memory",
            MODELS[2]: "Existing CausalHistoryTrajectoryModel GRU64(use_controls=True), direct history/future commands",
            MODELS[3]: "Frozen expanded September ActuatorAwareTrajectoryModel GRU64(use_controls=False), drive/tail gated residuals",
        },
        checkpoint_sha256={MODELS[3]: OURS_SHA}, ours_checkpoint=str(OURS.relative_to(ROOT)),
        seed_label=17, actual_training_seeds={MODELS[1]: {"initialization": 17}, MODELS[2]: {"initialization": 17}, MODELS[3]: original["seed"]},
        history_steps=26, nominal_history_span_s=.5, nominal_dt_s=.02, horizons_s=list(HORIZONS), primary_horizon_s=.5,
        timing_contract="5/10/25/50 native steps, nominal 0.1/0.2/0.5/1.0 s; integrate actual logged dt, no resampling; timing CSVs give actual durations",
        normalization={"source": "frozen Ours buffers; bitwise float32 verified by fitting TRAIN only", "values": {k: v.tolist() for k, v in stats.items()}},
        checkpoint_rule="last epoch of predeclared schedule; no validation selection",
        ours_training_protocol=original,
        metrics={"endpoint_vector_RMSE": "sqrt(mean_windows(sum_xyz(error**2))) per flight; no division by 3",
                 "attitude_error_deg": "RMS of sign-invariant quaternion geodesic angles per flight",
                 "macro": "equal-flight mean of per-flight RMSE; SD across flights ddof=1",
                 "increments": "(pred(t+h)-truth(t0))-(truth(t+h)-truth(t0)); exactly endpoint v/omega error",
                 "uncertainty": "single replicate; no seed SD, no window-based confidence interval or significance claim"},
        validation_origin_rule="all 2582 registered 1s windows, identical origins for every model/horizon; no error filtering",
        representative_rule="lexicographically sort validation (log_id,segment_id,start_sample_in_segment), choose index floor(N/2) BEFORE any predictions",
        future_known="logged motor,left,right,rudder controls at t0...t49, zero-order hold; dt",
        future_forbidden="all future measured physical states, wind, phase, frequency; labels only",
        fairness_limitations=["Frozen Ours trains backbone 40 epochs then freezes it for 25 actuator epochs; generic baselines have different trainable sets",
                              "Ours stage-two actuator regularization has no exact generic-model counterpart",
                              "Ours seeds are 17 then 29; replicate label 17 is not a claim of all-stage seed 17",
                              "B1 vs B2 also differs in capacity/activation; not a pure history-only ablation",
                              "Logged future commands were generated by a feedback controller; conditional replay is not causal intervention evidence"],
        sealed_test_status="This run opens only explicit train/validation Parquet files. Sealed/reserved names are manifest metadata only; no test predictions/results accessed.",
        sealed_test_opened_this_run=False, source_sha256=verify_sources(),
    )
    pd.DataFrame(prepared["coverage"]).to_csv(OUT / "flight_coverage.csv", index=False)
    write_json(OUT / "audit.json", {"checks": prepared["checks"], "parameter_counts": counts,
                                   "baseline_reuse": {"B0": "direct", "B1": "reuse MLP structure, new rollout adapter/retrain", "B2": "direct existing controlled GRU/retrain", "B3": "frozen reuse"},
                                   "expected_training": ["B1_MLP", "B2_StandardGRU"],
                                   "cost_reference": "Ours recorded 1827.4 s total; approximately 15-30 min per new baseline on cuda:1; pilot only"})
    if not (OUT / "protocol.json").exists():
        write_json(OUT / "protocol.json", protocol)
    return protocol


class _FrequencyLossAdapter(torch.nn.Module):
    """Training-only adapter: existing stage-two frequency loss, no fake actuator penalty."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def control_regularization_loss(self, **kwargs):
        return next(self.parameters()).new_zeros(())


def train_baseline(name, stats, prepared, original, schedule, device):
    checkpoint = ART / f"{name}.pt"
    model = build_model(name, stats)
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (saved["schedule"] != schedule or saved["identity"] != prepared["identity"]
                or saved.get('source_sha256') != verify_sources()):
            raise ValueError("baseline checkpoint protocol mismatch")
        model.load_state_dict(saved["state_dict"], strict=True)
        return model.eval()
    stages = [("base", 40, 17, .0003, False)]
    if schedule == "matched-budget":
        stages.append(("continuation", 25, 29, .0005, True))
    for stage, epochs, seed, lr, frequency in stages:
        stage_path = ART / f"{name}_{stage}.pt"
        if stage_path.exists():
            saved = torch.load(stage_path, map_location="cpu", weights_only=False)
            if (saved["identity"] != prepared["identity"] or saved["schedule"] != schedule
                    or saved.get('source_sha256') != verify_sources()):
                raise ValueError("stage checkpoint mismatch")
            model.load_state_dict(saved["state_dict"], strict=True)
            continue
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        trained = _FrequencyLossAdapter(model) if frequency else model
        def callback(_, history):
            history.to_csv(ART / f"{name}_{stage}_history.csv", index=False)
            status("training", model=name, training_stage=stage, **history.iloc[-1].to_dict())
        trained, history = train_increment_stage(
            trained, prepared["batches"]["train"], scales=original["scales"], weights=original["weights"],
            device=device, epochs=epochs, seed=seed, learning_rate=lr, actuator=frequency,
            steps=50, batch_size=256, callback=callback,
        )
        model = trained.model if frequency else trained
        torch.save(dict(state_dict=model.state_dict(), identity=prepared["identity"],
                        schedule=schedule, completed_stage=stage, source_sha256=verify_sources()), stage_path)
    torch.save(dict(state_dict=model.state_dict(), identity=prepared["identity"], schedule=schedule,
                    source_sha256=verify_sources()), checkpoint)
    return model.eval()


def causality_check(model, batch, device):
    # A shallow batch copy with distinct future truth arrays: no original mutation.
    poisoned_truth = copy.deepcopy(batch.trajectory.truth)
    for value in vars(poisoned_truth).values():
        value[:, 1:] = np.nan
    poisoned = replace(batch, trajectory=replace(batch.trajectory, truth=poisoned_truth))
    indices = np.arange(min(3, len(batch.trajectory.window_ids)))
    model.to(device).eval()
    with torch.inference_mode():
        p, _ = _model_call(model, batch, indices, use_history=True, rollout_steps=50, device=torch.device(device))
        other, _ = _model_call(model, poisoned, indices, use_history=True, rollout_steps=50, device=torch.device(device))
        short, _ = _model_call(model, batch, indices, use_history=True, rollout_steps=25, device=torch.device(device))
    for a, b, c in zip(p, other, short, strict=True):
        if not torch.equal(a, b) or not torch.equal(a[:, :26], c):
            raise ValueError("future-label poisoning or prefix parity failed")
    model.cpu()
    return dict(future_label_poisoning_bitwise_equal=True, prefix_500ms_bitwise_equal=True)


def evaluate(models, prepared, device, *, out=OUT, artifact_root=ART):
    batch = prepared["batches"]["validation"]
    origins = prepared["origins"]["validation"]
    ordered = origins.sort_values(["log_id", "segment_id", "start_sample_in_segment"])
    representative = int(ordered.index[len(ordered) // 2])
    checks, rows, examples = {}, [], {}
    for name, model in models.items():
        status("evaluating", model=name)
        if name == MODELS[0]:
            prediction = model.rollout(batch.trajectory.initial_state(), batch.trajectory.controls, batch.trajectory.dt_s)
            checks[name] = dict(analytic_constant_twist=True)
        else:
            checks[name] = causality_check(model, batch, device)
            prediction = predict_history_trajectory_model(model, batch, use_history=True, batch_size=128, device=device)
        frame = endpoint_metrics(prediction, batch.trajectory, model=name)
        rows.append(frame)
        checks[name].update(n_origins=len(origins), endpoint_rows=len(frame), finite=True,
                            max_quaternion_norm_error=float(np.max(np.abs(np.linalg.norm(prediction.quaternion_nb, axis=-1)-1))))
        np.savez_compressed(artifact_root / f"{name}_predictions.npz", **vars(prediction),
                            window_ids=batch.trajectory.window_ids.astype(str))
        for key, value in vars(prediction).items():
            examples[f"{name}_{key}"] = value[representative, :26]
    for key, value in vars(batch.trajectory.truth).items():
        examples[f"truth_{key}"] = value[representative, :26]
    examples["time_s"] = np.r_[0, np.cumsum(batch.trajectory.dt_s[representative, :25])]
    np.savez_compressed(out / "representative_prediction.npz", **examples)
    write_json(out / "representative_selection.json", dict(
        rule="fixed middle index after lexicographic origin identity sorting, independent of all model errors",
        sorted_index=len(origins)//2, source_index=representative,
        window_id=str(origins.iloc[representative].window_id),
        log_id=str(origins.iloc[representative].log_id),
        origin_timestamp_us=int(origins.iloc[representative].origin_timestamp_us)))
    rows = pd.concat(rows, ignore_index=True)
    expected = set(batch.trajectory.window_ids)
    for _, g in rows.groupby(["model", "horizon_s"]):
        if len(g) != len(expected) or set(g.window_id) != expected:
            raise ValueError("unequal evaluation origins")
    rows.to_csv(artifact_root / "per_origin.csv", index=False)
    per_flight, summary, aggregate = aggregate_flights(rows)
    per_flight.to_csv(out / "per_flight.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    aggregate.to_csv(out / "aggregate.csv", index=False)
    write_json(out / "sanity_checks.json", dict(data=prepared["checks"], models=checks,
                same_origins_all_models_horizons=True, normalization_train_only_reproduced=True,
                sealed_test_opened_this_run=False, checkpoint_unchanged=file_hash(OURS)==OURS_SHA))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["audit", "preflight", "train", "run"], default="audit")
    parser.add_argument("--model", choices=list(MODELS[1:3]))
    parser.add_argument("--schedule", choices=["matched-budget", "backbone-only"])
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    if args.phase in ('run','train') and not args.schedule:
        parser.error("run requires an explicit --schedule decision")
    if args.phase == 'train' and not args.model:
        parser.error('train worker requires --model')
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    try:
        prepared, ours, stats, original = prepare()
        if args.phase == 'train':
            train_baseline(args.model,stats,prepared,original,args.schedule,args.device)
            status('training_complete',model=args.model)
            return
        protocol = audit(prepared, ours, stats, original)
        if args.phase == "audit":
            status("audit_complete", counts=prepared["checks"], parameter_counts=protocol["parameter_counts"])
            return
        if args.phase == "preflight":
            smoke=ART/'preflight_untrained_NOT_RESULTS'
            smoke.mkdir(exist_ok=True)
            models={MODELS[0]:ConstantTwistPredictor(),
                    **{name:build_model(name,stats) for name in MODELS[1:3]}, MODELS[3]:ours}
            evaluate(models,prepared,args.device,out=smoke,artifact_root=smoke)
            status('preflight_complete_untrained_baselines',sealed_test_opened=False)
            return
        if (OUT / "summary.csv").exists():
            raise FileExistsError("completed results exist; refusing overwrite")
        torch.ones(1, device=args.device)
        protocol.update(status="pilot_running", schedule=args.schedule, device=args.device,
                        training_schedule=dict(base_epochs=40, continuation_epochs=25 if args.schedule=="matched-budget" else 0,
                            batch_size=256, optimizer="AdamW", learning_rates=[.0003, .0005] if args.schedule=="matched-budget" else [.0003],
                            weight_decay=1e-5, gradient_clip_norm=5, rollout_steps=50,
                            sampling_seeds=[17,29] if args.schedule=="matched-budget" else [17],
                            objective="existing trajectory_rollout_loss + frozen two-step increment weights/scales; continuation adds 0.2*frequency_MSE; no actuator regularizer"))
        for name in MODELS[1:3]:
            protocol["actual_training_seeds"][name]["stage_sampling"] = protocol["training_schedule"]["sampling_seeds"]
        write_json(OUT / "protocol.json", protocol)
        # Independent baseline workers share the otherwise idle GPU 1. Each
        # retains deterministic sampling and its own optimizer/RNG/checkpoint.
        processes=[]
        for name in MODELS[1:3]:
            log=(ART/f'{name}_training.log').open('a')
            env=dict(os.environ,PAPER_BASELINE_STATUS_NAME=f'{name}_status')
            command=[sys.executable,str(Path(__file__).resolve()),'--phase','train',
                     '--model',name,'--schedule',args.schedule,'--device',args.device]
            worker=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            processes.append((name,worker,log))
        failures=[]
        for name,worker,log in processes:
            code=worker.wait();log.close()
            if code:failures.append((name,code))
        if failures:
            raise RuntimeError(f'baseline training failed: {failures}')
        models = {MODELS[0]: ConstantTwistPredictor()}
        for name in MODELS[1:3]:
            models[name] = train_baseline(name, stats, prepared, original, args.schedule, args.device)
            protocol["checkpoint_sha256"][name] = file_hash(ART / f"{name}.pt")
        models[MODELS[3]] = ours
        evaluate(models, prepared, args.device)
        if file_hash(OURS) != OURS_SHA:
            raise ValueError("Ours checkpoint changed during benchmark")
        protocol["status"] = "seed17_pilot_complete"
        protocol["source_sha256"] = verify_sources()
        write_json(OUT / "protocol.json", protocol)
        status("pilot_complete", sealed_test_opened=False)
    except BaseException as error:
        status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
