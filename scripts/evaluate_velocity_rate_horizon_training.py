"""Matched validation of 200/500/1000ms-trained models through two seconds."""
import argparse
import json
from pathlib import Path
import sys
import time
import subprocess
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_velocity_rate_long_horizon as frozen
from train_velocity_rate_horizons import OUT, CACHE, OLD, LONG, verify_archives, setup
from train_velocity_rate_learning import batch
from system_identification.models.velocity_rate_gru import VelocityRateGRU, errors
from system_identification.models.control_events import CANONICAL_CONTROL_EVENTS
from system_identification.data.velocity_rate_dataset import write_json, file_hash
from verify_velocity_rate_learning import independent_errors

METRICS = frozen.METRICS
HORIZONS = [500, 1000, 2000]
SEEDS = [17, 29, 43]


def label(horizon, model):
    return "Kinematic Baseline" if model == "Kinematic Baseline" else f"H{horizon}-{model}"


def load_endpoints():
    frames = []
    for path in sorted((LONG / "metrics").glob("*_endpoints.parquet")):
        f = pd.read_parquet(path)
        f = f[f.horizon_ms.isin(HORIZONS)].copy()
        f["training_horizon_ms"] = 200 if f.model.iloc[0] != "Kinematic Baseline" else 0
        f["condition"] = label(int(f.training_horizon_ms.iloc[0]), f.model.iloc[0])
        frames.append(f)
    for path in sorted((OUT / "metrics").glob("*_endpoints.parquet")):
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


@torch.no_grad()
def evaluate(horizon):
    device = setup()
    # Validate the reused cache against the prior data audit; no dataset is rebuilt.
    old_audit = json.loads((LONG / "data_audit.json").read_text())
    assert file_hash(frozen.CACHE / "validation.npz") == old_audit["packed"]["cache_sha256"]
    old_archive = json.loads((LONG / "archive_manifest.json").read_text())
    index_path = frozen.CACHE / "common_indices.npy"
    assert file_hash(index_path) == old_archive["local_cache_artifacts"][str(index_path.relative_to(ROOT))]["sha256"]
    data = frozen.load_data(device)
    meta = pd.read_parquet(LONG / "common_windows.parquet")
    packed_meta = pd.read_parquet(LONG / "validation_windows.parquet")
    assert packed_meta.iloc[np.load(index_path)].window_id.tolist() == meta.window_id.tolist()
    times = data["dt"].cumsum(1).cpu().numpy()
    mask = data["future_mask"].cpu().numpy()
    example_meta = pd.read_csv(LONG / "example_windows.csv")
    choices = meta.reset_index().set_index("window_id").loc[example_meta.window_id, "index"].to_numpy()
    n = len(meta)
    (OUT / "metrics").mkdir(exist_ok=True)
    examples = {key: data[key][choices].cpu().numpy() for key in ("initial", "target", "dt", "future_mask")}
    native_frames, diagnostics, checks, runtimes, checkpoints = [], [], [], [], {}
    oracle = torch.cat([frozen.oracle(data["initial"][ix], data["target"][ix], data["dt"][ix])
                        for ix in torch.arange(n, device=device).split(2048)])
    for model_name in ("GRU-NoControl", "GRU-Control"):
        for seed in SEEDS:
            tick = time.time()
            name = label(horizon, model_name)
            path = OUT / f"train_{horizon}ms" / model_name / f"seed_{seed}/best.pt"
            expected = json.loads((path.parent / "summary.json").read_text())["checkpoint_sha256"]
            assert file_hash(path) == expected
            checkpoints[str(path.relative_to(ROOT))] = expected
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            model = VelocityRateGRU(checkpoint["normalization"], control=model_name == "GRU-Control").to(device)
            model.load_state_dict(checkpoint["model"])
            model.eval()
            torch.cuda.reset_peak_memory_stats(device)
            pred = torch.cat([model(batch(data, ix)) for ix in torch.arange(n, device=device).split(2048)])
            assert torch.isfinite(pred).all()
            er = errors(pred, data["target"]).cpu().numpy()
            er[..., 2] *= 180 / np.pi
            frame = frozen.endpoint_frame(er, times, meta, model_name, seed)
            frame = frame[frame.horizon_ms.isin(HORIZONS)].copy()
            frame["training_horizon_ms"], frame["condition"] = horizon, name
            slug = f"{name}_seed_{seed}"
            frame.to_parquet(OUT / f"metrics/{slug}_endpoints.parquet", index=False)
            np.savez_compressed(CACHE / f"{slug}_native_errors.npz", errors=er, mask=mask, actual_time_s=times)
            i, k = np.where(mask)
            native = pd.DataFrame(dict(flight_id=meta.flight_id.to_numpy()[i], native_step=k+1,
                                       actual_time_s=times[i, k]))
            for j, metric in enumerate(METRICS):
                native[metric] = er[i, k, j].astype(float)
            group = native.groupby(["flight_id", "native_step"])
            g = group[[*METRICS, "actual_time_s"]].mean().reset_index()
            g["n_windows"] = group.size().to_numpy()
            g["model"], g["condition"], g["seed"], g["training_horizon_ms"] = model_name, name, seed, horizon
            native_frames.append(g)
            delta = errors(pred, oracle).cpu().numpy()
            delta[..., 2] *= 180 / np.pi
            d = frozen.endpoint_frame(delta, times, meta, model_name, seed)
            d = d[d.horizon_ms.isin(HORIZONS)]
            d = d.groupby(["model", "seed", "horizon_ms", "flight_id"])[["position_m", "attitude_deg"]].mean().reset_index()
            d["condition"], d["training_horizon_ms"] = name, horizon
            diagnostics.append(d)
            examples[slug] = pred[choices].cpu().numpy()
            selected = np.linspace(0, n-1, 128, dtype=int)
            independent = independent_errors(pred[selected].cpu().numpy().astype(float),
                                              data["target"][selected].cpu().numpy().astype(float))
            difference = np.max(abs(independent-er[selected]), axis=(0, 1))
            assert difference.max() < .0002
            checks.append(dict(condition=name, seed=seed, scipy_max_difference=difference.tolist(),
                               quaternion_norm_max_error=(torch.linalg.vector_norm(pred[..., 6:10], dim=-1)-1).abs().max().item()))
            runtimes.append(dict(condition=name, seed=seed, seconds=time.time()-tick,
                                 peak_gpu_bytes=torch.cuda.max_memory_allocated(device), finite=True))
            print(f"evaluated {slug}: {time.time()-tick:.1f}s", flush=True)
    pd.concat(native_frames).to_csv(OUT / f"native_steps_{horizon}ms.csv", index=False)
    pd.concat(diagnostics).to_csv(OUT / f"dynamics_integral_{horizon}ms.csv", index=False)
    np.savez_compressed(OUT / f"trajectory_examples_{horizon}ms.npz", **examples)
    write_json(OUT / f"evaluation_{horizon}ms.json", dict(
        control_event_policy=CANONICAL_CONTROL_EVENTS,
        windows=n, flights=meta.flight_id.nunique(), device=str(device),
        checkpoints=checkpoints, numerical_checks=checks, runtime=runtimes,
        future_truth_feedback=False, test_opened=False))


def summarize():
    endpoints = load_endpoints()
    flight = endpoints.groupby(["condition", "model", "training_horizon_ms", "seed", "horizon_ms", "flight_id"])[METRICS].mean().reset_index()
    flight.to_csv(OUT / "per_flight.csv", index=False)
    seeds = flight.groupby(["condition", "seed", "horizon_ms"])[METRICS].mean().reset_index()
    seeds.to_csv(OUT / "per_seed.csv", index=False)
    rows = []
    for (condition, h), g in flight.groupby(["condition", "horizon_ms"]):
        f = g.groupby("flight_id")[METRICS].mean()
        s = seeds[(seeds.condition == condition) & (seeds.horizon_ms == h)]
        w = endpoints[(endpoints.condition == condition) & (endpoints.horizon_ms == h)].groupby("window_id")[METRICS].mean()
        for metric in METRICS:
            rows.append(dict(condition=condition, horizon_ms=h, metric=metric,
                             flight_mean=f[metric].mean(), flight_std=f[metric].std(ddof=1),
                             seed_std=s[metric].std(ddof=1), window_mean=w[metric].mean(),
                             window_std=w[metric].std(ddof=1), window_p95=w[metric].quantile(.95),
                             window_p99=w[metric].quantile(.99), n_windows=len(w), n_flights=len(f)))
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    pairs = []
    candidates = [label(h, m) for h in (200, 500, 1000) for m in ("GRU-NoControl", "GRU-Control")]
    comparisons = [("Kinematic Baseline", c) for c in candidates]
    comparisons += [(label(h, "GRU-NoControl"), label(h, "GRU-Control")) for h in (200, 500, 1000)]
    comparisons += [(label(a, m), label(b, m)) for a, b in ((200, 500), (200, 1000), (500, 1000))
                    for m in ("GRU-NoControl", "GRU-Control")]
    rng = np.random.default_rng(20261020)
    for h in HORIZONS:
        f = flight[flight.horizon_ms == h].groupby(["condition", "flight_id"])[METRICS].mean()
        for ref, candidate in comparisons:
            for metric in METRICS:
                a = f.loc[ref, metric]
                b = f.loc[candidate, metric].reindex(a.index)
                delta = (b-a).to_numpy()
                ci = np.quantile(delta[rng.integers(len(delta), size=(5000, len(delta)))].mean(1), [.025, .975])
                pairs.append(dict(horizon_ms=h, reference=ref, candidate=candidate, metric=metric,
                                  improvement_pct=100*(a.mean()-b.mean())/a.mean(), delta=delta.mean(),
                                  better_flights=int((delta < 0).sum()), bootstrap_low=ci[0], bootstrap_high=ci[1]))
    pd.DataFrame(pairs).to_csv(OUT / "paired_comparisons.csv", index=False)
    # Descriptive validation strata, shared by every model; never fitting/selection inputs.
    thresholds = pd.read_parquet(LONG / "common_windows.parquet").control_change.quantile([1/3, 2/3]).tolist()
    endpoints["control_change_group"] = pd.cut(endpoints.control_change,
                                             [-np.inf, *thresholds, np.inf], labels=["low", "medium", "high"]).astype(str)
    groups = []
    for field in ("review", "high_dynamics", "mode_change", "control_change_group"):
        window_counts = endpoints.groupby(["condition", "horizon_ms", field], observed=True).window_id.nunique()
        g = endpoints.groupby(["condition", "seed", "horizon_ms", field, "flight_id"], observed=True)[METRICS].mean().reset_index()
        g = g.groupby(["condition", "horizon_ms", field, "flight_id"], observed=True)[METRICS].mean().reset_index()
        for (condition, h, value), f in g.groupby(["condition", "horizon_ms", field], observed=True):
            for metric in METRICS:
                groups.append(dict(condition=condition, horizon_ms=h, grouping=field, group=value, metric=metric,
                                   flight_mean=f[metric].mean(), flight_std=f[metric].std(ddof=1), n_flights=len(f),
                                   n_windows=int(window_counts.loc[(condition, h, value)])))
    pd.DataFrame(groups).to_csv(OUT / "group_metrics.csv", index=False)
    observation = pd.read_parquet(LONG / "metrics/observation_integration.parquet")
    observation = observation[observation.horizon_ms.isin(HORIZONS)].copy()
    observation["control_change_group"] = pd.cut(observation.control_change,
        [-np.inf, *thresholds, np.inf], labels=["low", "medium", "high"]).astype(str)
    obs_rows = []
    for field in ("review", "high_dynamics", "mode_change", "control_change_group"):
        f = observation.groupby(["horizon_ms", field, "flight_id"])[["position_m", "attitude_deg"]].mean().reset_index()
        for (h, value), g in f.groupby(["horizon_ms", field]):
            for metric in ("position_m", "attitude_deg"):
                obs_rows.append(dict(horizon_ms=h, grouping=field, group=value, metric=metric,
                                     flight_mean=g[metric].mean(), flight_std=g[metric].std(ddof=1), n_flights=len(g)))
    pd.DataFrame(obs_rows).to_csv(OUT / "observation_group_metrics.csv", index=False)
    write_json(OUT / "group_definition.json", dict(
        control_change="L2 range of commands over history and 2s future event tape, descriptive validation terciles",
        thresholds=thresholds, model_input=False, checkpoint_selection=False))
    budgets = []
    for h, source in [(200, OLD), (500, OUT / "train_500ms"), (1000, OUT / "train_1000ms")]:
        for model in ("GRU-NoControl", "GRU-Control"):
            for seed in SEEDS:
                rec = json.loads((source / model / f"seed_{seed}/summary.json").read_text())
                rec["training_horizon_ms"] = h
                budgets.append(rec)
    pd.DataFrame(budgets).to_csv(OUT / "training_budget.csv", index=False)
    native = pd.read_csv(LONG / "native_steps_per_flight.csv")
    native["training_horizon_ms"] = np.where(native.model == "Kinematic Baseline", 0, 200)
    native["condition"] = [label(h, m) for h, m in zip(native.training_horizon_ms, native.model)]
    pd.concat([native, *[pd.read_csv(OUT / f"native_steps_{h}ms.csv") for h in (500, 1000)]],
              ignore_index=True).to_csv(OUT / "native_steps_per_flight.csv", index=False)
    print("matched endpoint, flight, seed and condition summaries saved", flush=True)


def verify():
    archives = verify_archives()
    for source in (OLD, LONG):
        for name, rec in json.loads((source / "archive_manifest.json").read_text()).get("source_files", {}).items():
            assert file_hash(ROOT / name) == rec["sha256"], name
    meta = pd.read_parquet(LONG / "common_windows.parquet")
    expected_windows = set(meta.window_id)
    endpoints = load_endpoints()
    counts = endpoints.groupby(["condition", "seed", "horizon_ms"]).size()
    assert (counts == 35946).all() and len(counts) == 57  # baseline + 18 checkpoints, three horizons
    for _, f in endpoints.groupby(["condition", "seed", "horizon_ms"]):
        assert set(f.window_id) == expected_windows
    summary = pd.read_csv(OUT / "summary.csv")
    worst = 0.
    for (condition, h), f in endpoints.groupby(["condition", "horizon_ms"]):
        independent = f.groupby(["flight_id", "seed"])[METRICS].mean().groupby("flight_id").mean().mean()
        reported = summary[(summary.condition == condition) & (summary.horizon_ms == h)].set_index("metric").loc[METRICS, "flight_mean"]
        worst = max(worst, float(abs(independent-reported).max()))
    assert worst < 1e-11
    # Check ordering, training budgets, identical loss scales and paired minibatch protocol.
    configs = [json.loads((OUT / f"train_{h}ms/config.json").read_text()) for h in (500, 1000)]
    for key in ("seeds", "batch_size", "max_epochs", "patience", "learning_rate", "weight_decay", "loss_weights"):
        assert configs[0][key] == configs[1][key]
    assert (OUT / "train_500ms/normalization.json").read_bytes() == (OUT / "train_1000ms/normalization.json").read_bytes()
    for split in ("train", "validation"):
        a = pd.read_parquet(OUT / f"train_500ms/{split}_windows.parquet")
        b = pd.read_parquet(OUT / f"train_1000ms/{split}_windows.parquet")
        assert a.equals(b)
    for horizon in (500, 1000):
        folder = OUT / f"train_{horizon}ms"
        provenance = json.loads((folder / "training_provenance.json").read_text())
        for name, expected in provenance["source_sha256"].items():
            assert file_hash(ROOT / name) == expected, name
        assert file_hash(folder / "config.json") == provenance["configuration_sha256"]
        assert file_hash(folder / "normalization.json") == provenance["normalization_sha256"]
        for model in ("GRU-NoControl", "GRU-Control"):
            for seed in SEEDS:
                run = folder / model / f"seed_{seed}"
                records = pd.read_json(run / "training.jsonl", lines=True)
                rec = json.loads((run / "summary.json").read_text())
                assert len(records) == rec["epochs"]
                assert int(records.updates.iloc[-1]) == rec["updates"] == 106*rec["epochs"]
                assert np.isfinite(records.select_dtypes(include=np.number).to_numpy()).all()
                assert file_hash(run / "best.pt") == rec["checkpoint_sha256"]
    write_json(OUT / "verification.json", dict(status="passed", matched_windows=35946, flights=27,
        model_seed_conditions=19, endpoint_groups=len(counts), aggregation_max_difference=worst,
        prior_archives_verified=archives, test_opened=False, paired_training_protocol=True))
    print("independent aggregation, cohorts, protocols and prior archives verified", flush=True)


def archive():
    assert json.loads((OUT / "verification.json").read_text())["status"] == "passed"
    sources = [ROOT / "scripts" / name for name in (
        "train_velocity_rate_horizons.py", "evaluate_velocity_rate_horizon_training.py",
        "report_velocity_rate_horizon_training.py")]
    sources += [ROOT / "tests/test_velocity_rate_horizon_training.py"]
    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for path in sources:
        committed = subprocess.check_output(["git", "show", f"{code_commit}:{path.relative_to(ROOT)}"])
        assert committed == path.read_bytes(), "Commit the reviewed implementation before archiving"
    source_records = {str(p.relative_to(ROOT)): dict(sha256=file_hash(p), bytes=p.stat().st_size) for p in sources}
    run = json.loads((OUT / "run_manifest.json").read_text())
    run.update(implementation_commit=code_commit, source_files=source_records,
               execution_provenance="training started atop base_commit with source hashes recorded; those exact bytes are now committed")
    write_json(OUT / "run_manifest.json", run)
    files = {str(p.relative_to(OUT)): dict(sha256=file_hash(p), bytes=p.stat().st_size)
             for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "archive_manifest.json"}
    caches = {str(p.relative_to(ROOT)): dict(sha256=file_hash(p), bytes=p.stat().st_size)
              for p in sorted(CACHE.glob("*.npz"))}
    write_json(OUT / "archive_manifest.json", dict(
        schema="velocity_rate_horizon_training_v1", test_sealed=True, control_ready=False,
        implementation_commit=code_commit, source_files=source_records, files=files,
        file_count=len(files), total_bytes=sum(x["bytes"] for x in files.values()),
        local_cache_artifacts=caches, reused_frozen_validation=str(LONG.relative_to(ROOT))))
    for name, rec in files.items():
        assert file_hash(OUT / name) == rec["sha256"]
    print(f"archived and rechecked {len(files)} files, {sum(x['bytes'] for x in files.values())} bytes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["evaluate", "summarize", "verify", "archive"])
    parser.add_argument("--horizon", type=int, choices=[500, 1000])
    args = parser.parse_args()
    if args.stage == "evaluate":
        assert args.horizon is not None
        evaluate(args.horizon)
    else:
        globals()[args.stage]()
