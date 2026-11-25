"""Align all flights across arms and bootstrap the same flights in each seed."""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEEDS = (17, 29, 43)
METRICS = ("v05", "w05", "v1", "w1", "v2", "w2")


def summarize(study):
    state = json.loads((study / "state.json").read_text())
    arms = {"fixed": state["baseline"]}
    for item in state["experiments"].values():
        if item.get("result"):
            arm = {"frequency-delay": "held", "measured-delay": "measured"}[item["hypothesis"]["hypothesis_id"]]
            arms[arm] = item
    arrays = {}
    flights = None
    training = {}
    for arm, item in arms.items():
        run = study / item["attempts"][-1]["directory"]
        for seed in SEEDS:
            folder = run / f"seed-{seed}"
            scores = json.loads((folder / "output/flight_mse.json").read_text())
            training[f"{arm}-{seed}"] = json.loads((folder / "training.json").read_text())
            for variant, row in scores.items():
                name = "corrected" if variant == "corrected" else f"{variant}-{arm}"
                for metric in METRICS:
                    ids = sorted(row[metric])
                    if flights is None:
                        flights = ids
                    assert flights == ids and len(flights) == 39
                    a = np.array([row[metric][f] for f in ids])
                    key = (name, metric)
                    by_seed = arrays.setdefault(key, {})
                    if seed in by_seed:
                        np.testing.assert_allclose(a, by_seed[seed], atol=1e-7, rtol=1e-6)
                    by_seed[seed] = a
    draws = np.random.default_rng(17).integers(0, len(flights), (10000, len(flights)))
    means, boots, data, scores = {}, {}, {}, {}
    for (name, metric), values in arrays.items():
        a = np.stack([values[s] for s in SEEDS])
        data[name, metric] = a
        indiv = np.sqrt(a.mean(1))
        means[name, metric] = indiv
        scores.setdefault(name, {})[metric] = {"mean": float(indiv.mean()), "seeds": indiv.tolist()}
        boots[name, metric] = np.sqrt(a[:, draws].mean(2)).mean(0)
    contrasts = {}
    ordered = ["corrected", *[f"residual-{a}" for a in arms], *[f"direct-{a}" for a in arms]]
    for ref, candidate in combinations(ordered, 2):
        if ref.startswith("residual") and candidate.startswith("direct"):
            continue
        pairs = {}
        for key in METRICS:
            d = means[ref, key] - means[candidate, key]
            paired = boots[ref, key] - boots[candidate, key]
            pairs[key] = dict(gain=float(d.mean()), seed_gains=d.tolist(),
                              ci95=np.quantile(paired, [.025, .975]).tolist(),
                              wins=int(((data[ref, key] - data[candidate, key]).mean(0) > 0).sum()))
        contrasts[f"{ref} minus {candidate}"] = pairs
    return dict(study=state["id"], arms={a: x["id"] for a,x in arms.items()}, seeds=SEEDS,
                flights=flights, metrics=scores, gain_comparisons=contrasts, training=training)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("study", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.study), indent=2) + "\n")
