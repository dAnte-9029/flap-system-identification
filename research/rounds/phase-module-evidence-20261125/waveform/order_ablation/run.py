"""Matched two-by-two order-phase ablation, using the second-round fitting and evaluation routines."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "coherence"
ARMS = ("both_wrong", "second_true", "third_true", "full")
METRICS = ("v05", "w05", "v1", "w1", "v2", "w2")


def load_round_two():
    # The prior runner looks up these two functions when it builds the normal equations
    # and scores the original batches. Use a private module instance so file inputs and
    # the archived runner stay unchanged.
    spec = importlib.util.spec_from_file_location("frozen_waveform_round_two", PRIOR / "run.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    get_features = runner.features
    runner.MOMENTS = ("conditional",)

    def order_features(coef, time, theta, hist, u0, pooled, shift, params, include_legacy=False):
        args = hist, u0, pooled, shift, params
        right = get_features(coef, time, theta, *args)["conditional"]
        wrong = get_features(coef, time, theta + shift[:, None, None], *args)["conditional"]
        # Two sin/cos orientations × three temporal envelopes per order, in this sequence.
        return {"full":right, "both_wrong":wrong,
                "second_true":torch.cat([right[..., :6], wrong[..., 6:]], -1),
                "third_true":torch.cat([wrong[..., :6], right[..., 6:]], -1)}

    def forecasts(base, designs, time, weights, legacy):
        return {arm: base + torch.einsum("bqcxm,cxm->bqcx", designs[arm], weights[arm])
                for arm in ARMS}
    runner.features = order_features
    runner.output = forecasts
    runner.ARMS = ARMS
    return runner


def compare(archive, current, keys, metrics):
    old = archive[archive.arm == "conditional"].set_index(keys)[list(metrics)].sort_index()
    new = current[current.arm == "full"].set_index(keys)[list(metrics)].sort_index()
    assert old.index.equals(new.index) and not new.isna().any().any()
    return float(abs(old - new).to_numpy().max())


@torch.inference_mode()
def run(root, device):
    runner = load_round_two()
    wave = runner.wave
    with np.load(PRIOR / "clock.npz") as z:
        clock = {"conditional": torch.as_tensor(z["conditional"], dtype=torch.float32, device=device)}
    train, train_ids, train_meta = wave.load(root, "train", device)
    dev, dev_ids, dev_meta = wave.load(root, "validation", device)
    assert set(train_ids).isdisjoint(dev_ids) and (train_meta["flights"], dev_meta["flights"]) == (121, 39)
    cache = wave.native_cache(root, device)
    eps, native, stationarity, weight_error = [], [], {}, {}
    for seed in wave.SEEDS:
        state = torch.load(wave.PRIOR / f"readout/original_runs/incumbent{seed}/corrected.pt",
                           weights_only=True, map_location=device)
        model = wave.FlightPredictor(state["mean"],state["std"]).to(device).eval()
        model.load_state_dict(state)
        weights, stationarity[seed] = runner.train(model, train, clock)
        arrays = {k:w.cpu().numpy() for k,w in weights.items()}
        assert all(w.size == 72 for w in arrays.values())
        np.savez(HERE / f"weights_{seed}.npz", **arrays)
        with np.load(PRIOR / f"weights_{seed}.npz") as z:
            weight_error[seed] = float(np.max(abs(z["conditional"] - arrays["full"])))
        eps.extend(runner.endpoints(model, dev, dev_ids, clock, weights, None, seed))
        native.append(runner.native(model, dev, cache, clock, weights, None, seed))
        print("finished seed", seed, flush=True)
    ep = pd.DataFrame(eps)
    native = pd.concat(native,ignore_index=True)
    ep.to_csv(HERE / "flight_mse.csv", index=False)
    native.to_csv(HERE / "native_flight_mse.csv", index=False)
    ep_error = compare(pd.read_csv(PRIOR/"flight_mse.csv"), ep, ["seed","flight"], METRICS)
    ns_error = compare(pd.read_csv(PRIOR/"native_flight_mse.csv"), native, ["seed","flight","ch","period"],
                       ["measurement","higher"])
    assert max(ep_error, ns_error, max(weight_error.values())) < 1e-5
    report = dict(train=train_meta, validation=dev_meta, weight_count=72, normal_equation=stationarity,
                  archived_full_max_abs_diff=dict(weights=weight_error,endpoint_mse=ep_error,native_mse=ns_error),
                  endpoints=wave.paired(ep,METRICS),native={
                      f"{ch}_{period}":wave.paired(group,["measurement","higher"])
                      for (ch,period),group in native.groupby(["ch","period"])})
    (HERE / "results.json").write_text(json.dumps(report,indent=2) + "\n")


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root",type=Path,default=Path("/home/zn/flap-system-identification"))
    ap.add_argument("--device",default="cuda:0")
    args=ap.parse_args()
    torch.set_num_threads(2)
    run(args.data_root, args.device)
