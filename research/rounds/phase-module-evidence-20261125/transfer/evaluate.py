"""Add the seed-17 waveform correction to an independent predictor, without fitting."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
WAVE = HERE.parent / "waveform"
sys.path.insert(0, str(WAVE))
from experiment import FlightPredictor, METRICS, PRIOR, design, load
sys.path.insert(0, str(WAVE / "coherence"))
from inference import HarmonicTransport, causal_conditions


@torch.inference_mode()
def evaluate(source_project, prediction_file, output, backbone):
    data, flights, _ = load(source_project, "validation", "cpu")
    zpath = source_project / "outputs/autopilot_flight_dynamics_20261124/validation.npz"
    with np.load(zpath) as original:
        ids, expected_time, expected_flights = original["window"], original["time"], original["flight"]
    # Require explicit identity even if the benchmark happened to emit the original order.
    with np.load(prediction_file) as saved:
        index = pd.Index(saved["window_id"])
        assert index.is_unique and len(index) == len(ids)
        order = index.get_indexer(ids)
        assert (order >= 0).all()
        predicted_delta = saved["predicted_delta"][order]
        np.testing.assert_array_equal(saved["time"][order], expected_time)
        np.testing.assert_array_equal(saved["flight_id"][order], expected_flights)
    assert len(ids) == 44524 and len(set(flights)) == 39
    if predicted_delta.shape == (len(ids), 8, 2, 3):
        predicted_delta = predicted_delta[:, [1, 3, 7]]
    assert predicted_delta.shape == (len(ids), 3, 2, 3) and np.isfinite(predicted_delta).all()
    state = torch.load(PRIOR / "readout/original_runs/incumbent17/corrected.pt",
                       weights_only=True, map_location="cpu")
    original_predictor = FlightPredictor(state["mean"], state["std"])
    original_predictor.load_state_dict(state)
    with np.load(WAVE / "weights_17.npz") as z:
        weights = torch.tensor(z["full"])
    head = HarmonicTransport.from_files(17)
    with np.load(WAVE / "coherence/clock.npz") as z:
        sham_head = HarmonicTransport(torch.from_numpy(z["sham"]), head.readout_coefficients)
    predictions = {k: [] for k in (backbone, "full", "origin", "wrong_phase", "second_round", "second_sham")}
    for start in range(0, len(ids), 768):
        sl = slice(start, start + 768)
        t = data["time"][sl][:, [1, 3, 7]]
        pool = data["pooled"][sl][:, [1, 3, 7]]
        angle = original_predictor.clocks(data["hist"][sl], pool, data["u0"][sl], t)[..., -1]
        base = torch.from_numpy(predicted_delta[sl])
        y = data["y"][sl][:, [1, 3, 7]]
        cond = causal_conditions(data["hist"][sl], pool, data["u0"][sl], t)
        for name, module in (("second_round", head), ("second_sham", sham_head)):
            pred = base + module(data["coef"][sl], t, angle, cond)
            predictions[name].append((pred - y).square().sum(-1).reshape(-1, 6).numpy())
        for name, arm in ((backbone, None), ("full", "full"), ("origin", "origin"), ("wrong_phase", "sham")):
            pred = base
            if arm is not None:
                z = design(data["coef"][sl], t, angle, data["shift"][sl], arm)
                pred = base + torch.einsum("bqcxm,cxm->bqcx", z, weights[..., :z.shape[-1]])
            predictions[name].append((pred - y).square().sum(-1).reshape(-1, 6).numpy())
    distinct = np.unique(flights)
    mask = data["mask"][:, [1, 3, 7]].reshape(-1, 6).numpy().astype(bool)
    mse = {}
    for name, values in predictions.items():
        a = np.concatenate(values)
        mse[name] = {m: {f: float(a[(flights == f) & mask[:, j], j].mean(dtype=np.float64)) for f in distinct}
                     for j, m in enumerate(METRICS)}
    draws = np.random.default_rng(17).integers(0, len(distinct), (10000, len(distinct)))
    summary = {}
    for m in METRICS:
        arrays = {k: np.fromiter(v[m].values(), float) for k, v in mse.items()}
        scores = {k: float(np.sqrt(v.mean())) for k, v in arrays.items()}
        contrasts = {}
        for chosen, refs in (("full", (backbone, "origin", "wrong_phase")), ("second_round", (backbone, "second_sham"))):
            gains = {}
            for k in refs:
                v, full = arrays[k], arrays[chosen]
                paired = np.sqrt(v[draws].mean(1)) - np.sqrt(full[draws].mean(1))
                gains[k] = dict(gain=scores[k] - scores[chosen],
                                ci95=np.quantile(paired, [.025, .975]).tolist(),
                                improved=int((v > full).sum()))
            contrasts[chosen] = gains
        summary[m] = dict(rmse=scores, gains=contrasts)
    output.write_text(json.dumps(dict(seed=17, backbone=backbone, flight_mse=mse, summary=summary), indent=2) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("prediction_file", type=Path)
    ap.add_argument("--backbone", choices=["tcn", "ode"], default="tcn")
    ap.add_argument("--source-project", type=Path, default=Path("/home/zn/flap-system-identification"))
    ap.add_argument("--output", type=Path, default=HERE / "result.json")
    opts = ap.parse_args()
    torch.set_num_threads(3)
    evaluate(opts.source_project, opts.prediction_file, opts.output, opts.backbone)
