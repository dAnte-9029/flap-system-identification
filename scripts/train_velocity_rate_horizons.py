"""Paired 500ms/1s autonomous training; frozen data, model and test boundary."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_velocity_rate_learning as prep
import train_velocity_rate_learning as training
from system_identification.data.velocity_rate_dataset import file_hash, write_json

OUT = ROOT / "experiments/velocity_rate_gru64_horizon_training_v1"
CACHE = ROOT / "artifacts/velocity_rate_gru64_horizon_training_v1"
OLD = ROOT / "experiments/velocity_rate_gru64_round1"
LONG = ROOT / "experiments/velocity_rate_gru64_long_horizon_v1"
HORIZONS = (500, 1000)


def verify_archives():
    verified = {}
    for source in (OLD, LONG):
        manifest = source / "archive_manifest.json"
        for name, entry in json.loads(manifest.read_text())["files"].items():
            assert file_hash(source / name) == entry["sha256"], (source, name)
        verified[str(source.relative_to(ROOT))] = file_hash(manifest)
    return verified


def crop_horizon(data, counts):
    """Keep registered endpoints, including the occasional 26th/51st sample."""
    counts = np.asarray(counts, dtype=np.int64)
    assert len(counts) == len(data["initial"]) and (counts > 0).all()
    assert (counts <= data["future_mask"].sum(1)).all()
    width = int(counts.max())
    mask = np.arange(width)[None, :] < counts[:, None]
    result = dict(data)
    for key in ("target", "dt", "events", "event_length"):
        value = data[key][:, :width].copy()
        if key == "target":
            # Repeat the last valid target for padding, never used by the loss.
            last = value[np.arange(len(counts)), counts - 1]
            value = np.where(mask[..., None], value, last[:, None])
        else:
            value[~mask] = 1 if key == "event_length" else 0
        result[key] = value
    result["future_mask"] = mask
    return result


def prepare():
    OUT.mkdir(exist_ok=False)
    CACHE.mkdir(exist_ok=False)
    old_archives = verify_archives()
    config = json.loads((OLD / "config.json").read_text())
    config.update(horizon_us=1000000, expected_windows={"train": 107920, "validation": 36332})
    root, audit = prep.resolve(config)
    prep.EXP, prep.CACHE = OUT, CACHE
    audit["partitions"] = {}
    for split in ("train", "validation"):
        tick = time.time()
        audit["partitions"][split] = prep.pack(root, split, config)
        meta = pd.read_parquet(OUT / f"{split}_windows.parquet")
        all_windows = pd.read_parquet(root / f"windows_{split}.parquet")
        short = all_windows.loc[
            (all_windows.sampling_group == "main_50hz")
            & (all_windows.history_us == 500000)
            & (all_windows.horizon_us == 500000)
        ].set_index(["flight_id", "origin_sample"])
        matched = short.loc[list(zip(meta.flight_id, meta.origin_sample))]
        assert (matched.start_sample.to_numpy() == meta.start_sample.to_numpy()).all()
        meta["count_500ms"] = matched.future_count.to_numpy()
        meta["count_1000ms"] = meta.future_count
        meta.to_parquet(OUT / f"{split}_windows.parquet", index=False)
        print(f"packed {split}: {len(meta)} shared origins, {time.time()-tick:.1f}s", flush=True)
    assert set(pd.read_parquet(OUT / "train_windows.parquet").flight_id).isdisjoint(
        pd.read_parquet(OUT / "validation_windows.parquet").flight_id)
    for horizon in HORIZONS:
        folder = OUT / f"train_{horizon}ms"
        folder.mkdir()
        run = copy.deepcopy(config)
        run.update(horizon_us=horizon * 1000,
                   loss_scale_policy="shared train-only RMS baseline residual over all native steps through 1s",
                   input_output_normalization="frozen from original train-only 200ms experiment",
                   paired_cohort="identical registered 1s origins, also present in 500ms windows",
                   checkpoint_selection=f"validation equal-flight loss through {horizon}ms only",
                   evaluation_horizons_ms=[500, 1000, 2000])
        write_json(folder / "config.json", run)
        for split in ("train", "validation"):
            pd.read_parquet(OUT / f"{split}_windows.parquet").to_parquet(
                folder / f"{split}_windows.parquet", index=False)
    audit.update(prior_archives=old_archives, test_opened=False)
    write_json(OUT / "data_audit.json", audit)
    write_json(OUT / "run_manifest.json", dict(
        base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        dataset_id=config["dataset_id"], dataset_manifest_sha256=audit["manifest_sha256"],
        seeds=config["seeds"], train_horizons_ms=list(HORIZONS),
        evaluation_horizons_ms=[500, 1000, 2000], test_opened=False, control_ready=False,
        model_source_sha256=file_hash(ROOT / "src/system_identification/models/velocity_rate_gru.py"),
        future_feedback=False, future_control="recorded closed-loop command replay; Control only",
        phase_forecast=config["phase_forecast"], prior_archives=old_archives,
        historical_200ms_comparison="reference only: original training cohort and loss scales differ",
    ))


def numpy_data(split, horizon):
    assert split in ("train", "validation")
    tick = time.time()
    archive = np.load(CACHE / f"{split}.npz")
    data = {key: archive[key] for key in archive.files}
    meta = pd.read_parquet(OUT / f"{split}_windows.parquet")
    result = crop_horizon(data, meta[f"count_{horizon}ms"].to_numpy())
    print(f"loaded/cropped {split} {horizon}ms in {time.time()-tick:.1f}s", flush=True)
    return result


def gpu_data(split, horizon, device):
    return {key: torch.from_numpy(value).to(device)
            for key, value in numpy_data(split, horizon).items()}


def setup():
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    return torch.device("cuda:0")


def normalize():
    device = setup()
    data = gpu_data("train", 1000, device)
    config = json.loads((OUT / "train_1000ms/config.json").read_text())
    fitted = training.fit_normalization(data, config)
    norm = json.loads((OLD / "normalization.json").read_text())
    norm["loss_scales"] = fitted["loss_scales"]
    norm["source"] = "input/output frozen from original train; shared loss scales from current 1s train only"
    write_json(OUT / "normalization.json", norm)
    # Make the relative weighting at both horizons visible before training.
    from system_identification.models.velocity_rate_gru import baseline, errors
    components = {}
    for horizon in HORIZONS:
        count = pd.read_parquet(OUT / "train_windows.parquet")[f"count_{horizon}ms"].to_numpy()
        sums = torch.zeros(4, device=device)
        for ix in torch.arange(len(count), device=device).split(2048):
            width = int(count.max())
            pred = baseline(data["initial"][ix], data["dt"][ix, :width])
            e = errors(pred, data["target"][ix, :width])
            mask = torch.arange(width, device=device)[None] < torch.as_tensor(count, device=device)[ix, None]
            sums += ((e / torch.tensor(norm["loss_scales"], device=device)).square()
                     * mask[..., None]).sum((0, 1))
        components[str(horizon)] = (sums / int(count.sum())).tolist()
        write_json(OUT / f"train_{horizon}ms/normalization.json", norm)
    write_json(OUT / "loss_scale_audit.json", dict(
        order=["position_m", "velocity_mps", "attitude_rad", "angular_rate_radps"],
        loss_scales=norm["loss_scales"], weights=config["loss_weights"],
        train_baseline_mean_scaled_square=components, validation_used=False))
    print(json.dumps(components), flush=True)


def run(horizon, smoke_only=False):
    device = setup()
    training.EXP = OUT / f"train_{horizon}ms"
    config = json.loads((training.EXP / "config.json").read_text())
    norm = json.loads((training.EXP / "normalization.json").read_text())
    tick = time.time()
    td, vd = gpu_data("train", horizon, device), gpu_data("validation", horizon, device)
    torch.cuda.synchronize()
    if smoke_only or not (training.EXP / "smoke.json").exists():
        write_json(training.EXP / "runtime.json", dict(
            gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
            load_seconds=time.time()-tick, threads=4, tf32=True, cudnn_deterministic=True,
            gpu_resident_data_bytes=sum(v.numel()*v.element_size() for d in (td, vd) for v in d.values())))
        training.smoke(config, norm, td, device)
        if smoke_only:
            return
    assert (training.EXP / "smoke.json").exists()
    write_json(training.EXP / "training_provenance.json", dict(
        base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256={str(p.relative_to(ROOT)): file_hash(p) for p in (
            Path(__file__), ROOT / "scripts/train_velocity_rate_learning.py",
            ROOT / "scripts/prepare_velocity_rate_learning.py",
            ROOT / "src/system_identification/models/velocity_rate_gru.py")},
        configuration_sha256=file_hash(training.EXP / "config.json"),
        normalization_sha256=file_hash(training.EXP / "normalization.json")))
    for seed in config["seeds"]:
        for model in ("GRU-NoControl", "GRU-Control"):
            training.train(config, norm, td, vd, device, model, seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "normalize", "smoke", "train"])
    parser.add_argument("--horizon", type=int, choices=HORIZONS)
    args = parser.parse_args()
    if args.stage in ("prepare", "normalize"):
        globals()[args.stage]()
    else:
        assert args.horizon is not None
        run(args.horizon, smoke_only=args.stage == "smoke")
