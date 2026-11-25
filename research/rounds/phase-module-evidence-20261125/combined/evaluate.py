import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from predictor import HERE, ROUND, WingbeatPredictor, prior
sys.path.insert(0, str(ROUND / "waveform"))
from experiment import load, paired, METRICS, PRIOR, SEEDS


@torch.inference_mode()
def score(source, device):
    d, flights, _ = load(source, "validation", device)
    rows = []
    assert len(flights) == 44524 and len(set(flights)) == 39
    mask = d["mask"][:, [1, 3, 7]].reshape(-1, 6).cpu().numpy()
    for seed in SEEDS:
        new = WingbeatPredictor.from_seed(seed, device)
        params = torch.load(PRIOR / f"readout/original_runs/incumbent{seed}/corrected.pt",
                            map_location=device, weights_only=True)
        legacy = prior.FlightPredictor(params["mean"], params["std"]).to(device)
        legacy.load_state_dict(params)
        values = {k: [] for k in ("corrected", "direction", "transport", "full")}
        for start in range(0, len(flights), 768):
            sl = slice(start, start + 768)
            args = [d[k][sl] for k in ("hist", "pooled", "u0", "time")]
            old = legacy(*args)
            low, high = new.contributions(*args, d["coef"][sl])
            for arm, pred in (("corrected", old), ("direction", low), ("transport", old + high), ("full", low + high)):
                sq = (pred - d["y"][sl]).square().sum(-1)[:, [1, 3, 7]]
                values[arm].append(sq.reshape(-1, 6).cpu().numpy())
        for arm, batches in values.items():
            sq = np.concatenate(batches)
            for fid in np.unique(flights):
                row = {m: float(sq[(flights == fid) & mask[:, j], j].mean(dtype=np.float64))
                       for j, m in enumerate(METRICS)}
                rows.append(dict(arm=arm, seed=seed, flight=fid, **row))
    frame = pd.DataFrame(rows)
    frame.to_csv(HERE / "flight_mse.csv", index=False)
    result = paired(frame, METRICS)
    # Compare both standalone arms against their independently evaluated serialized predictions.
    wf = pd.read_csv(ROUND / "waveform/coherence/flight_mse.csv")
    target = wf[wf.arm == "conditional"].set_index(["seed", "flight"])[list(METRICS)].sort_index()
    measured = frame[frame.arm == "transport"].set_index(["seed", "flight"])[list(METRICS)].sort_index()
    np.testing.assert_allclose(target, measured, atol=1e-6, rtol=1e-6)
    comparator = json.loads((ROUND / "coherence/flight_mse.json").read_text())
    for row in frame[frame.arm == "direction"].itertuples():
        reference = comparator[str(row.seed)]["direction"]
        np.testing.assert_allclose([getattr(row, m) for m in METRICS],
                                   [reference[m][row.flight] for m in METRICS],
                                   atol=2e-6, rtol=2e-6)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("/home/zn/flap-system-identification"))
    ap.add_argument("--device", default="cpu")
    opts = ap.parse_args()
    torch.set_num_threads(3)
    (HERE / "results.json").write_text(json.dumps(score(opts.source, opts.device), indent=2) + "\n")
