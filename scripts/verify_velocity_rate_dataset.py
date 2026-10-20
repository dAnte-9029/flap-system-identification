"""Independent exhaustive data integrity checks; read-only unless --report supplied."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from system_identification.data.velocity_rate_dataset import (
    STATE, TARGET, CONTROL, file_hash, interval_overlap, write_json,
)


def verify(root):
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    for name, expected in manifest["artifact_sha256"].items():
        assert file_hash(root / name) == expected, f"artifact mismatch: {name}"
    provenance = {r["flight_id"]: r for r in json.loads((root / "flight_provenance.json").read_text())}
    verified_rows = verified_windows = 0
    for split in ("train", "validation", "test"):
        samples = pd.read_parquet(root / f"samples_{split}.parquet")
        windows = pd.read_parquet(root / f"windows_{split}.parquet")
        assert not any("acceleration" in c or "derivative" in c for c in samples.columns)
        natives = {name: pd.read_parquet(root / f"native_{name}_{split}.parquet")
                   for name in ("vehicle_local_position", "vehicle_attitude", "vehicle_angular_velocity",
                                "actuator_motors", "actuator_servos", "encoder_count", "flap_frequency")}
        groups = {name: dict(tuple(frame.groupby("flight_id", sort=False))) for name, frame in natives.items()}
        wg = dict(tuple(windows.groupby("flight_id", sort=False)))
        for fid, frame in samples.groupby("flight_id", sort=False):
            frame = frame.reset_index(drop=True)
            record = provenance[fid]
            assert record["split"] == split
            local = groups["vehicle_local_position"][fid].set_index("timestamp_us").loc[frame.timestamp_us]
            np.testing.assert_array_equal(frame.state_measurement_us, local.timestamp_sample_us)
            np.testing.assert_array_equal(frame[STATE[:6]], local[["x", "y", "z", "vx", "vy", "vz"]])
            np.testing.assert_array_equal(frame[TARGET[:6]], frame[STATE[:6]])
            for topic, prefix, fields, columns in (
                ("vehicle_attitude", "attitude", [f"q[{i}]" for i in range(4)], STATE[6:10]),
                ("vehicle_angular_velocity", "rate", [f"xyz[{i}]" for i in range(3)], STATE[10:13]),
            ):
                raw = groups[topic][fid]
                for target in (False, True):
                    p = "target_" if target else ""
                    key = f"{p}{prefix}_{'measurement' if target else 'publication'}_us"
                    native_key = "timestamp_sample_us" if target else "timestamp_us"
                    selected = raw.set_index(native_key).loc[frame[key], fields].to_numpy()
                    np.testing.assert_array_equal(frame[[p+c for c in columns]].to_numpy(), selected)
                    if not target:
                        assert (frame[key] <= frame.timestamp_us).all()
            for topic, prefix in (("encoder_count", "encoder"), ("flap_frequency", "frequency")):
                raw = groups[topic][fid].set_index("timestamp_us")
                selected = raw.loc[frame[f"{prefix}_source_timestamp_us"]]
                np.testing.assert_array_equal(frame[f"{prefix}_publication_us"], selected.availability_us)
                assert (frame[f"{prefix}_publication_us"] <= frame.timestamp_us).all()
                assert (frame[f"{prefix}_source_timestamp_us"] < frame[f"{prefix}_publication_us"]).all()
                if not record["firmware_verified"]:
                    assert not frame["valid_encoder" if prefix == "encoder" else "valid_frequency"].any()
            # Validate all future u[k] at actual measurement times, independently of loader.
            for topic, fields in (("actuator_motors", ["control[0]"]),
                                  ("actuator_servos", ["control[0]", "control[1]", "control[2]"])):
                raw = groups[topic][fid]
                ts = raw.timestamp_us.to_numpy()
                idx = np.searchsorted(ts, frame.state_measurement_us, side="right") - 1
                w = wg[fid]
                used = np.zeros(len(frame)+1, np.int64)
                np.add.at(used, w.origin_sample.to_numpy(int), 1)
                np.add.at(used, w.end_sample.to_numpy(int), -1)
                used = np.cumsum(used[:-1]) > 0
                assert (idx[used] >= 0).all()
                ages = frame.state_measurement_us.to_numpy()[used] - ts[idx[used]]
                assert ((ages >= 0) & (ages <= manifest["config"]["control_age_us"])).all()
                assert np.isfinite(raw[fields].to_numpy()[idx[used]]).all()
            w = wg[fid]
            a, o, b = (w[c].to_numpy(int) for c in ("start_sample", "origin_sample", "end_sample"))
            bad = np.r_[0, np.cumsum(~frame.valid_core.to_numpy())]
            assert ((bad[b+1]-bad[a]) == 0).all()
            assert (frame.segment_id.to_numpy()[a] == frame.segment_id.to_numpy()[b]).all()
            tm = frame.state_measurement_us.to_numpy()
            assert (tm[o+1] > frame.timestamp_us.to_numpy()[o]).all()
            np.testing.assert_allclose((tm[b]-tm[o])*1e-6, w.actual_horizon_s, atol=1e-12)
            ex = np.array(record["exclude_intervals_us"], dtype=np.int64).reshape(-1, 2)
            assert not interval_overlap(w.support_start_us, w.support_end_us, ex).any()
            rev = np.array(record["review_intervals_us"], dtype=np.int64).reshape(-1, 2)
            np.testing.assert_array_equal(interval_overlap(w.support_start_us, w.support_end_us, rev), w.review)
            verified_rows += len(frame)
            verified_windows += len(w)
        print(f"verified {split}: {len(samples)} rows / {len(windows)} windows", flush=True)
    return dict(status="passed", verified_rows=verified_rows, verified_windows=verified_windows,
                artifact_hashes_verified=len(manifest["artifact_sha256"]),
                checks=["native_state_identity", "q_w_source_identity", "publisher_upper_bounds",
                        "all_future_start_controls", "all_window_masks", "EXCLUDE", "REVIEW",
                        "native_horizons", "no_acceleration_labels", "unknown_firmware_phase_masks"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", type=Path)
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    result = verify(args.dataset)
    if args.report:
        write_json(args.report, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
