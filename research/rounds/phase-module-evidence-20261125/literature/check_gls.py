"""Check the author's GLS against an independent three-column weighted fit."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def check(source):
    spec = importlib.util.spec_from_file_location("official_gls", source)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)

    rng = np.random.default_rng(17)
    times = np.sort(rng.uniform(2.13, 5.41, 83))
    relative = times - times.min()
    sigma = rng.uniform(0.02, 0.10, len(times))
    omega = 2 * np.pi * 3.75
    values = 1.4 + 0.85 * np.cos(omega * relative) - 0.32 * np.sin(omega * relative)
    values += rng.normal(0, sigma)
    frequencies = np.linspace(2.75, 4.75, 65)
    result = official.Gls((times, values, sigma), freq=frequencies,
                          fbeg=frequencies.min(), fend=frequencies.max())
    max_coefficient_error = 0.
    max_power_error = 0.
    weighted_mean = np.average(values, weights=1 / sigma**2)
    offset_sse = np.square((values - weighted_mean) / sigma).sum()
    for k, frequency in enumerate(frequencies):
        phase = 2 * np.pi * frequency * relative
        design = np.column_stack([np.cos(phase), np.sin(phase), np.ones(len(times))])
        weights, *_ = np.linalg.lstsq(design / sigma[:, None], values / sigma, rcond=None)
        fitted = np.array([result._a[k], result._b[k], result._off[k] + result._Y])
        max_coefficient_error = max(max_coefficient_error, np.max(abs(weights - fitted)))
        ref_power = 1 - np.square((values - design @ weights) / sigma).sum() / offset_sse
        max_power_error = max(max_power_error, abs(ref_power - result.p[k]))
    assert max_coefficient_error < 1e-10 and max_power_error < 1e-10
    return dict(n_samples=len(times), n_frequencies=len(frequencies),
                max_coefficient_error=max_coefficient_error, max_normalized_power_error=max_power_error,
                recovered_peak_hz=result.best['f'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Path to pinned mzechmeister/GLS/python/gls.py')
    print(json.dumps(check(parser.parse_args().source), indent=2))
