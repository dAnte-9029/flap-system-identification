"""Fixed optimizer, sample order and flight-equal native-time validation for both studies."""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
RELATIVE = Path("experiments/autopilot_flight_dynamics")
INPUTS = ("hist", "pooled", "u0", "time")
NAMES = ("v05", "w05", "v1", "w1", "v2", "w2")
BATCH = 384


def load(part, *, monitor=False):
    if monitor:
        assert os.environ["RA_SPLIT"] == "train" and part == "validation"
    else:
        assert os.environ["RA_SPLIT"] == part
    assert part in ("train", "validation")
    meta = json.loads((HERE / "data.json").read_text())
    assert meta["dataset"] == os.environ["RA_DATA_VERSION"]
    rec = meta["split"][part]
    path = Path(rec["file"])
    with path.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == rec["sha256"]
    with np.load(path, allow_pickle=False) as f:
        batch = {k: torch.from_numpy(f[k]).to("cuda") for k in (*INPUTS, "y", "mask")}
        flights = f["flight"]
    assert len(flights) == rec["windows"] and len(np.unique(flights)) == rec["flights"]
    return batch, flights


def predict(model, data, index):
    return model(*(data[key][index] for key in INPUTS))


def preflight():
    from model import FlightPredictor
    from intervals import interval, prefix_grid
    b, _ = load("train")
    batch = {k: v[:16] for k, v in b.items()}
    hist = batch["hist"]
    model = FlightPredictor(hist.mean(0), hist.std(0).clamp_min(0.05)).to("cuda")
    pooled, time = prefix_grid(batch["pooled"], batch["time"])
    dt, control = interval(pooled, time, batch["u0"], 0, 2)
    assert dt.min() > 0 and torch.isfinite(control).all()
    pred = predict(model, batch, slice(None))
    assert pred.shape == batch["y"].shape and torch.isfinite(pred).all()
    modified = {**batch, "pooled": batch["pooled"].clone(), "time": batch["time"].clone()}
    modified["pooled"][:, 4:] += 1.0
    modified["time"][:, 4:] += 0.01
    torch.testing.assert_close(predict(model, modified, slice(None))[:, :4], pred[:, :4], rtol=0, atol=0)
    print("GPU", torch.cuda.get_device_name(), "train windows", len(b["hist"]), "suffix independence passed")


@torch.inference_mode()
def development_loss(model, data, weights, scale):
    model.eval()
    total = torch.zeros((), device=scale.device, dtype=torch.float64)
    for i in range(0, len(weights), 768):
        chunk = slice(i, i + 768)
        standardized = (predict(model, data, chunk) - data["y"][chunk]) / scale
        squared = standardized.square().sum(-1) * data["mask"][chunk]
        total += (squared.sum((1, 2)).double() * weights[chunk]).sum()
    model.train()
    return total.item()


def train():
    from model import FlightPredictor
    seed, steps = int(os.environ["RA_SEED"]), int(os.environ["RA_STEPS"])
    torch.manual_seed(seed)
    b, train_flights = load("train")
    dev, dev_flights = load("validation", monitor=True)
    assert set(train_flights).isdisjoint(dev_flights)
    members, index = np.unique(dev_flights, return_inverse=True)
    counts = np.bincount(index, weights=dev["mask"].sum((1, 2)).cpu().numpy())
    assert (counts > 0).all()
    weights = torch.from_numpy(1 / (3 * len(members) * counts[index])).to("cuda")
    rule = json.loads(os.environ["RA_TRAINING_COMPLETION"])
    assert rule["mode"] == "validation_plateau" and rule["metric"] == "dev_mse"
    hist, y, mask = b["hist"], b["y"], b["mask"].unsqueeze(-1)
    model = FlightPredictor(hist.mean(0), hist.std(0).clamp_min(0.05)).to("cuda")
    scale = ((y.square() * mask).sum((0, 1)) / mask.sum((0, 1))).sqrt().clamp_min(0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3,
                                                         threshold=0.001, threshold_mode="abs", min_lr=0.000125)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    initial = [p.detach().clone() for p in model.parameters()]
    history = []
    for step in range(1, steps + 1):
        ix = torch.randint(len(y), (BATCH,), generator=generator).to("cuda")
        pred = predict(model, b, ix)
        loss = (((pred - y[ix]) / scale).square() * mask[ix]).sum() / (3 * mask[ix].sum())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
        opt.step()
        losses.append(float(loss.detach()))
        if step % rule["eval_every"] == 0 or step == steps:
            score = development_loss(model, dev, weights, scale)
            assert np.isfinite(score)
            row = dict(step=step, dev_mse=score, train_mse=float(np.mean(losses[-50:])), lr=opt.param_groups[0]["lr"])
            history.append(row)
            print(json.dumps(row), flush=True)
            scheduler.step(score)
            from policy import on_monitor
            if on_monitor(history, model):
                break
    delta = sum((p.detach() - prev).square().sum() for p, prev in zip(model.parameters(), initial)).sqrt().item()
    info = dict(steps=step, samples=step * BATCH, validation_history=history, first50=float(np.mean(losses[:50])),
                train_loss=float(np.mean(losses[-50:])), parameter_change_l2=delta,
                parameters=sum(p.numel() for p in model.parameters()), gpu=torch.cuda.get_device_name(),
                cuda_visible=os.environ["CUDA_VISIBLE_DEVICES"])
    assert delta > 0 and np.isfinite(losses).all()
    torch.save(model.state_dict(), os.environ["RA_MODEL"])
    Path(os.environ["RA_TRAIN_INFO"]).write_text(json.dumps(info))
    print("Finished at step", step, "with parameter change", delta)


def evaluate():
    b, flights = load("validation")
    path = Path(os.environ["RA_MODEL_CODE"]) / RELATIVE / "model.py"
    spec = importlib.util.spec_from_file_location("fixed_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = torch.load(os.environ["RA_MODEL"], weights_only=True, map_location="cpu")
    model = module.FlightPredictor(state["mean"], state["std"]).to("cuda")
    model.load_state_dict(state)
    model.eval()
    with torch.inference_mode():
        pred = torch.cat([predict(model, b, slice(i, i + 768)) for i in range(0, len(flights), 768)])
    assert torch.isfinite(pred).all()
    sq = (pred - b["y"]).square().sum(-1)[:, [1, 3, 7]].reshape(-1, 6).cpu().numpy()
    valid = b["mask"][:, [1, 3, 7]].reshape(-1, 6).cpu().numpy()
    mse, metrics = {}, {}
    members = np.unique(flights)
    for i, name in enumerate(NAMES):
        mse[name] = {f: float(sq[(flights == f) & valid[:, i], i].mean()) for f in members}
        metrics[name] = float(np.sqrt(np.mean(list(mse[name].values()))))
    assert len(members) == 39 and np.isfinite(list(metrics.values())).all()
    Path(os.environ["RA_METRICS"]).write_text(json.dumps(metrics))
    (Path(os.environ["RA_OUTPUT"]) / "flight_mse.json").write_text(json.dumps(mse))
    print(metrics)


if __name__ == "__main__":
    torch.set_num_threads(2)
    {"preflight": preflight, "train": train, "evaluate": evaluate}[sys.argv[1]]()
