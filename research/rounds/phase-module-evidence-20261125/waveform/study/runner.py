"""Re-fit the registered H4 contrast and score it from fixed evaluator code."""
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
WAVE = HERE.parent
ROOT = HERE.parents[4]
SOURCE = Path("/home/zn/flap-system-identification")
sys.path.insert(0, str(ROOT / "experiments/conditional_waveform"))
from arm import PAIRING
sys.path.insert(0, str(WAVE / "coherence"))
from inference import HarmonicTransport, causal_conditions
import run as research

wave = research.wave


def restore(seed, device):
    state = torch.load(wave.PRIOR / f"readout/original_runs/incumbent{seed}/corrected.pt",
                       map_location=device, weights_only=True)
    base = wave.FlightPredictor(state["mean"], state["std"]).to(device)
    base.load_state_dict(state)
    return base.eval()


@torch.inference_mode()
def train(seed, device):
    data, flights, coverage = wave.load(SOURCE, "train", device)
    assert len(set(flights)) == 121
    with np.load(WAVE / "coherence/clock.npz") as f:
        moments = {k: torch.as_tensor(f[k], device=device).float() for k in research.MOMENTS}
    weights, check = research.train(restore(seed, device), data, moments)
    model = HarmonicTransport(moments[PAIRING], weights[PAIRING])
    torch.save(model.cpu().state_dict(), os.environ["RA_MODEL"])
    info = dict(steps=1, samples=int(data["mask"].sum() + data["mask_early"].sum()),
                normal_equation_relative_residual=check[PAIRING], coverage=coverage, coupling=PAIRING)
    assert check[PAIRING] < 1e-8
    Path(os.environ["RA_TRAIN_INFO"]).write_text(json.dumps(info) + "\n")


@torch.inference_mode()
def evaluate(seed, device):
    data, flights, _ = wave.load(SOURCE, "validation", device)
    assert len(set(flights)) == 39
    base = restore(seed, device)
    state = torch.load(os.environ["RA_MODEL"], map_location=device, weights_only=True)
    model = HarmonicTransport(state["moment_coefficients"], state["readout_coefficients"]).to(device)
    model.load_state_dict(state)
    errors = []
    for i in range(0, len(flights), 768):
        sl = slice(i, i + 768)
        old, theta = wave.prediction_parts(base, data, sl, "")
        t, pool, hist, u = (data[k][sl] for k in ("time", "pooled", "hist", "u0"))
        pred = old + model(data["coef"][sl], t, theta, causal_conditions(hist, pool, u, t))
        errors.append((pred - data["y"][sl]).square().sum(-1)[:, [1, 3, 7]].reshape(-1, 6).cpu().numpy())
    a = np.concatenate(errors)
    ok = data["mask"][:, [1, 3, 7]].reshape(-1, 6).cpu().numpy()
    rows = {m: {f: a[(flights == f) & ok[:, j], j].mean(dtype=np.float64)
                for f in np.unique(flights)} for j, m in enumerate(wave.METRICS)}
    (Path(os.environ["RA_OUTPUT"]) / "flight_mse.json").write_text(json.dumps(rows) + "\n")
    scores = {m: np.sqrt(np.mean(list(v.values()))) for m, v in rows.items()}
    Path(os.environ["RA_METRICS"]).write_text(json.dumps(scores) + "\n")


if __name__ == "__main__":
    torch.set_num_threads(2)
    device = "cuda:0"
    if sys.argv[1] == "preflight":
        assert torch.cuda.is_available() and PAIRING in ("sham", "conditional")
        with np.load(WAVE / "coherence/clock.npz") as f:
            assert f[PAIRING].shape == (45, 4)
    else:
        {"train": train, "evaluate": evaluate}[sys.argv[1]](int(os.environ["RA_SEED"]), device)
