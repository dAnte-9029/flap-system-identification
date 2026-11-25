"""Fit and score causal onset anchors on the three frozen flight predictors."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
PREVIOUS = PROJECT / "research/rounds/periodic-motion-20261125"
sys.path.insert(0, str(PREVIOUS / "readout"))
from model import FlightPredictor

ARMS = ("trend", "nominal", "encoder")
SEEDS = (17, 29, 43)
METRICS = ("v05", "w05", "v1", "w1", "v2", "w2")
PHASE_HZ = 3.8


def local_anchors(hist, main):
    """Estimate slow velocity/rate at decision time, retaining common observable columns."""
    batch, steps = hist.shape[:2]
    seconds = np.linspace(-.5, 0., steps)
    time = np.broadcast_to(seconds, (batch, steps))
    common = [np.ones_like(time), time * 2.]
    nominal = np.stack(common + [np.cos(2*np.pi*PHASE_HZ*time), np.sin(2*np.pi*PHASE_HZ*time)], -1)
    actual = np.stack(common + [hist[..., 14], hist[..., 13]], -1)
    designs = [nominal[..., :2], nominal, actual]
    past_phase = hist[..., 21] > 0
    masks = (past_phase & (hist[..., 16] > 0) & (hist[..., 18] > 0) & main[:, None],
             past_phase & (hist[..., 17] > 0))
    anchors = np.zeros((batch, len(ARMS), 2, 3), dtype=np.float32)
    usable = np.zeros((batch, 2), dtype=bool)
    for stream, mask in enumerate(masks):
        good = (mask.sum(1) >= 12) & past_phase[:, -1]
        physical = hist[..., 3*stream:3*stream+3]
        for arm, design in enumerate(designs):
            masked = design * mask[..., None]
            gram = masked.transpose(0, 2, 1) @ masked
            eig = np.linalg.eigvalsh(gram)
            ok = (eig[:, 0] > .0001 * eig[:, -1])
            good &= ok
            rhs = masked.transpose(0, 2, 1) @ physical
            coeffs = np.linalg.solve(np.where(ok[:, None, None], gram, np.eye(gram.shape[-1])), rhs)
            anchors[:, arm, stream] = coeffs[:, 0]
        usable[:, stream] = good
    return anchors, usable


def load_split(root, split, device):
    cache = root / "outputs"
    history_dir = cache / "history_slow_fast_20261119_cache"
    groups = pd.read_parquet(history_dir / f"{split}_windows.parquet",
                             columns=["window_id", "flight_id", "sampling_group"]).set_index("window_id", verify_integrity=True)
    history = np.load(history_dir / f"{split}_history.npy", mmap_mode="r")
    with np.load(cache / "autopilot_flight_dynamics_20261124" / f"{split}.npz") as z:
        idx = groups.index.get_indexer(z["window"])
        assert np.all(idx >= 0)
        flights = z["flight"]
        np.testing.assert_array_equal(groups.iloc[idx].flight_id, flights)
        data = {k: torch.from_numpy(z[k]).to(device) for k in ("hist", "time", "pooled", "u0", "y", "mask")}
    old = np.load(cache / "autopilot_periodic_motion_20261125" / f"{split}_features.npy")
    data["hist"] = torch.cat([data["hist"], torch.as_tensor(old, device=device)], 1)
    main = groups.iloc[idx].sampling_group.to_numpy() == "main_50hz"
    anchors, usable = zip(*(local_anchors(history[idx[i:i+768]], main[i:i+768])
                            for i in range(0, len(idx), 768)))
    anchor, usable = np.concatenate(anchors), np.concatenate(usable)
    np.testing.assert_array_equal(usable, old[:, -2:].astype(bool))
    data["anchor"] = torch.as_tensor(anchor, device=device)
    data["usable"] = torch.as_tensor(usable, device=device)
    coverage = {"flights": len(np.unique(flights)), "windows": len(idx),
                "main_50hz_windows": int(main.sum()), "available": usable.sum(0).tolist()}
    return data, flights, coverage


def time_basis(t):
    return torch.stack([-torch.expm1(-t/.1), t/2, (t/2)*(1-t/2)], -1)


def predictions(model, data, sl):
    hist = data["hist"][sl]
    time = data["time"][sl]
    slow, wave, origin_wave = model.components(hist, data["pooled"][sl], data["u0"][sl], time)
    origin = hist[:, 150:156].reshape(-1, 2, 3)
    slow_increment = slow - origin_wave
    base = slow_increment + wave
    absolute_slow = origin[:, None] + slow_increment
    discrepancy = data["anchor"][sl][:, :, None] - absolute_slow[:, None]
    discrepancy *= data["usable"][sl][:, None, None, :, None]
    return base, discrepancy, time_basis(time)


def fitted_gains(model, data):
    normal = torch.zeros(3, 2, 3, 3, 3, device=model.mean.device, dtype=torch.float64)
    cross = torch.zeros_like(normal[..., 0])
    for start in range(0, len(data["hist"]), 768):
        sl = slice(start, start + 768)
        base, discrepancy, basis = predictions(model, data, sl)
        residual = (data["y"][sl] - base).double()
        design = discrepancy.double()[..., None] * basis.double()[:, None, :, :, None]
        design = design * data["mask"][sl][:, None, :, :, None, None]
        normal += torch.einsum("bnqcxk,bnqcxj->ncxkj", design, design)
        cross += torch.einsum("bnqcxk,bqcx->ncxk", design, residual)
    gain = torch.linalg.solve(normal, cross[..., None]).squeeze(-1)
    err = ((normal @ gain.unsqueeze(-1)).squeeze(-1)-cross).abs().max() / cross.abs().max()
    assert err < 1e-10
    return gain, float(err)


def evaluate(model, data, flights, gains, seed):
    arms = ("corrected", *ARMS, "encoder_on_nominal", "instant_persistence")
    table = []
    tensors = []
    for start in range(0, len(flights), 768):
        sl = slice(start, start+768)
        base, delta, basis = predictions(model, data, sl)
        weights = torch.einsum("ncxk,bqck->bnqcx", gains.float(), basis)
        outputs = [base, *(base+weights[:,i]*delta[:,i] for i in range(3)),
                   base+weights[:,2]*delta[:,1], torch.zeros_like(base)]
        errors = torch.stack([(pred-data["y"][sl]).square().sum(-1) for pred in outputs], 1)
        tensors.append(errors[:, :, (1,3,7)].flatten(2).cpu().numpy())
    values = np.concatenate(tensors)
    mask = data["mask"][:, (1,3,7)].reshape(len(flights), 6).cpu().numpy()
    for fid in np.unique(flights):
        idx = flights == fid
        for arm_id, arm in enumerate(arms):
            row = {"flight": str(fid), "seed": seed, "arm": arm}
            for i, key in enumerate(METRICS):
                ok = idx & mask[:, i]
                row[key] = float(np.mean(values[ok, arm_id, i], dtype=np.float64))
            table.append(row)
    return table


def native(root, data, gains, seed):
    prior = root / "research/rounds/periodic-motion-20261125/diagnostics"
    sys.path.insert(0, str(prior))
    import evaluate as probe
    with np.load(prior / "reference.npz") as file:
        ref = dict(file)
    with np.load(prior / f"original_corrected{seed}_points.npz") as file:
        pred = dict(file)
    ix, stream = ref["window_idx"], ref["ch"]
    anchor = data["anchor"].cpu().numpy()[ix, :, stream]
    usable = data["usable"].cpu().numpy()[ix, stream]
    basis = time_basis(torch.tensor(ref["time"])).numpy()
    metrics = []
    for arm_id, arm in enumerate(ARMS):
        g = (basis[:, None, :] * gains[arm_id, stream]).sum(-1)
        corrected = pred["slow_abs"] + (anchor[:,arm_id]-pred["slow_abs"]) * usable[:,None]*g
        blocks = probe.block_values(ref, {**pred, "slow_abs": corrected})
        # Average beats in window, then windows in flight.
        def aggregate(frame):
            return frame.groupby(["ch","flight","window"]).mean(numeric_only=True).groupby(["ch", "flight"]).mean()
        overall = aggregate(blocks)
        later = aggregate(blocks[blocks.is_later])
        for (ch,f), row in overall.iterrows():
            metrics.append(dict(seed=seed, arm=arm, ch=int(ch), flight=str(ref["flight_ids"][f]),
                                absolute_slow=row.absolute_slow_error, measured=row.full_observation,
                                slow_change=later.loc[(ch,f)].slow_change_error,
                                persist_change=later.loc[(ch,f)].zero_slow_change))
    return metrics


def summaries(table):
    names = sorted(table.flight.unique())
    rng = np.random.default_rng(17)
    draws = rng.integers(0,len(names),size=(10000,len(names)))
    scores = {}
    rows = []
    for m in METRICS:
        def stack(arm):
            return table[table.arm==arm].pivot(index="seed",columns="flight",values=m).loc[list(SEEDS),names].to_numpy()
        means = {arm: stack(arm).mean(1)**.5 for arm in table.arm.unique()}
        scores[m] = {k: {"mean":float(v.mean()), "seeds":v.tolist()} for k,v in means.items()}
        for name in ("trend","nominal","encoder"):
            for other in ("corrected","nominal","trend","encoder_on_nominal"):
                if other==name or (name!="encoder" and other!="corrected"):
                    continue
                ref, candidate = stack(other), stack(name)
                gain = (ref[:,draws].mean(2)**.5-candidate[:,draws].mean(2)**.5).mean(0)
                rows.append(dict(metric=m, candidate=name, reference=other,
                                 gain=float((means[other]-means[name]).mean()),
                                 lo=float(np.quantile(gain,.025)),hi=float(np.quantile(gain,.975)),
                                 improved_flights=int((candidate.mean(0)<ref.mean(0)).sum())))
    return scores, rows


@torch.inference_mode()
def run(root, device):
    train, train_flights, a = load_split(root, "train", device)
    dev, val_flights, b = load_split(root, "validation", device)
    assert set(train_flights).isdisjoint(val_flights) and a["flights"]==121 and b["flights"]==39
    rows, diag, residual = [], [], {}
    for s in SEEDS:
        ckpt = PREVIOUS / f"readout/original_runs/incumbent{s}/corrected.pt"
        state = torch.load(ckpt, weights_only=True, map_location=device)
        model = FlightPredictor(state["mean"], state["std"]).to(device).eval()
        model.load_state_dict(state)
        g, residual[s] = fitted_gains(model, train)
        np.save(HERE / f"gain_seed{s}.npy", g.cpu().numpy())
        rows.extend(evaluate(model, dev, val_flights, g, s))
        diag.extend(native(root, dev, g.cpu().numpy(), s))
        print("finished seed",s,flush=True)
    flights, beats = pd.DataFrame(rows), pd.DataFrame(diag)
    flights.to_csv(HERE/"flight_mse.csv", index=False)
    beats.to_csv(HERE/"native_flight_mse.csv", index=False)
    scores, contrasts = summaries(flights)
    result = dict(train=a, validation=b, train_normal_equation_relative_error=residual,
                  metrics=scores,paired=contrasts)
    (HERE/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    print(scores["w2"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",type=Path,default=Path("/home/zn/flap-system-identification"))
    parser.add_argument("--device",default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args.data_root, args.device)
