"""Behavioral tests for native-time causality and full-window exclusion."""
from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from system_identification.data.velocity_rate_dataset import (
    CONTROL, STATE, action_replay, build_windows, flight_date, hold,
    interval_overlap, nearest_indices, paired_target, rotate_ned_to_body,
    relative_phase_from_total_count, VelocityRateDataset,
)
from system_identification.data.velocity_rate_build import prepare_split

ROOT = Path(__file__).resolve().parents[1]


def config():
    return yaml.safe_load((ROOT / "configs/data/velocity_rate_multistep_v1.yaml").read_text())


def native(times, values, field="value", sample=None):
    return pd.DataFrame(dict(timestamp_us=times,
                             timestamp_sample_us=sample if sample is not None else np.array(times)-10,
                             **{field: values}))


def test_publication_gate_does_not_use_early_measurement_of_late_message():
    topic = native([100, 220], [1, 999], sample=[90, 110])
    value, valid, timing = hold(topic, np.array([105, 200, 220]), ["value"], "x", 1000)
    np.testing.assert_equal(value[:, 0], [1, 1, 999])
    np.testing.assert_equal(timing["x_age_us"], [5, 100, 0])
    assert valid.all()


def test_serial_publisher_upper_bound_delays_encoder_feature():
    topic = native([100, 200, 300], [10, 20, 30], sample=[-1]*3)
    topic["availability_us"] = [200, 300, np.iinfo(np.int64).max]
    value, valid, timing = hold(topic, np.array([150, 250, 350]), ["value"], "encoder", 1000)
    assert not valid[0]
    np.testing.assert_equal(value[1:, 0], [10, 20])
    np.testing.assert_equal(timing["encoder_age_us"][1:], [150, 150])
    np.testing.assert_equal(timing["encoder_publication_us"][1:], [200, 300])


def test_targets_can_pair_later_measurement_but_history_cannot():
    topic = native([110, 210], [1, 2], sample=[100, 200])
    value, valid, timing = paired_target(topic, np.array([195]), ["value"], "q", 10)
    assert value[0, 0] == 2 and valid[0] and timing["target_q_offset_us"][0] == 5
    past, _, _ = hold(topic, np.array([195]), ["value"], "q", 1000)
    assert past[0, 0] == 1


def test_closed_interval_exclusion_between_samples():
    assert interval_overlap([100], [200], [[150, 160]])[0]
    assert interval_overlap([100], [200], [[200, 201]])[0]
    assert not interval_overlap([100], [200], [[201, 220]])[0]


def fixture_samples(step=20000, n=300):
    t = 1_000_000 + np.arange(n)*step
    return pd.DataFrame(dict(state_measurement_us=t, timestamp_us=t+500,
                             support_start_us=t-1000, support_end_us=t+1000,
                             segment_id=np.zeros(n, dtype=int), mode_change=False,
                             high_dynamics=False, valid_encoder=True, valid_frequency=True))


def fixture_record(group="main_50hz"):
    return dict(flight_id="fixture:1", split="train", sampling_group=group,
                exclude_intervals_us=[], review_intervals_us=[], reset_times_us=[])


def test_duration_windows_support_native_steps_and_keep_review_and_modes():
    s, r = fixture_samples(), fixture_record()
    s.loc[70, "mode_change"] = True
    r["review_intervals_us"] = [[2_350_000, 2_351_000]]
    w, _ = build_windows(s, r, config())
    assert w.mode_change.any() and w.review.any()
    primary = w[w.horizon_us == 100000]
    assert (primary.future_count == 5).all()
    assert (w.actual_history_s >= w.history_us/1e6).all()
    assert np.allclose(w.actual_horizon_s, w.horizon_us/1e6)


@pytest.mark.parametrize("kind", ["exclude", "reset"])
def test_entire_history_and_future_span_is_checked(kind):
    s, r = fixture_samples(), fixture_record()
    if kind == "exclude":
        r["exclude_intervals_us"] = [[2_310_001, 2_310_002]]
    else:
        r["reset_times_us"] = [2_310_001]
    w, rejects = build_windows(s, r, config())
    assert rejects[kind] > 0
    assert not ((w.support_start_us <= 2_310_001) & (w.support_end_us >= 2_310_001)).any()


def test_windows_never_bridge_segment_or_invalid_rows():
    s, r = fixture_samples(), fixture_record()
    s.loc[150, "segment_id"] = -1
    s.loc[151:, "segment_id"] = 1
    w, _ = build_windows(s, r, config())
    assert not ((w.start_sample <= 150) & (w.end_sample >= 150)).any()


def test_sparse_group_has_only_one_or_two_short_horizon_steps():
    w, _ = build_windows(fixture_samples(step=100000), fixture_record("sparse_10hz"), config())
    assert (w.loc[w.horizon_us == 100000, "future_count"] == 1).all()
    assert (w.loc[w.horizon_us == 200000, "future_count"] == 2).all()


def test_action_order_and_no_future_average_in_start_control():
    motor = native([80, 150, 250], [.1, .9, .4], "control[0]")
    servo = native([70, 160, 260], [.2, .8, .5], "control[0]")
    servo["control[1]"] = [-.3, -.4, -.5]
    servo["control[2]"] = [.6, .7, .8]
    replay = action_replay(motor, servo, np.array([100, 200, 300]), 1000)
    np.testing.assert_allclose(replay["step_control"], [[.1, .2, -.3, .6], [.9, .8, -.4, .7]])
    assert (replay["step_control_publication_us"] <= np.array([100, 200])[:, None]).all()
    assert 150 in replay["event_timestamp_us"] and 160 in replay["event_timestamp_us"]
    assert replay["event_timestamp_us"][-1] < 300


def test_quaternion_direction_and_original_phase_anchor():
    q = np.array([[np.sqrt(.5), 0, 0, np.sqrt(.5)]])
    np.testing.assert_allclose(rotate_ned_to_body(q, [[0, 1, 0]]), [[1, 0, 0]], atol=1e-12)
    unwrapped, wrapped = relative_phase_from_total_count([120, 4216, 8312], transmission_ratio=8)
    np.testing.assert_allclose(unwrapped, [0, np.pi/4, np.pi/2])
    assert wrapped[1] != 0  # taking a history slice must retain this initial phase


def test_bad_target_clock_and_frozen_test_loader_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="strictly increasing"):
        nearest_indices([1, 1], [1])
    with pytest.raises(ValueError, match="test is frozen"):
        VelocityRateDataset(tmp_path, split="test")


def test_recovered_date_groups_with_physical_date():
    assert flight_date("9.25-2数据/target_1723_reconstructed.ulg") == "2026-09-25"
    assert flight_date("2026.9.8/log_0_2026-9-8-05-42-48.ulg") == "2026-09-08"


def test_future_truth_mutation_cannot_change_model_inputs(tmp_path):
    from system_identification.data.velocity_rate_dataset import PHASE, TARGET
    n = 12
    times = 1_000_000 + np.arange(n)*20000
    columns = STATE + CONTROL + PHASE + TARGET + [
        f"target_velocity_body_m_s_{axis}" for axis in "xyz"]
    s = pd.DataFrame({c: np.arange(n, dtype=float) for c in columns})
    for c in ["valid_control", "valid_encoder", "valid_frequency", "target_position_velocity_valid",
              "target_attitude_valid", "target_rate_valid", "target_phase_valid", "target_frequency_valid"]:
        s[c] = True
    for c in ["state_age_us", *[f"{p}_age_us" for p in
                              ["attitude", "rate", "motor", "servo", "encoder", "frequency"]]]:
        s[c] = 100
    s["flight_id"] = "flight"
    s["timestamp_us"] = times + 100
    s["state_measurement_us"] = times
    s["relative_flap_phase_unwrapped_rad"] = np.arange(n)*.1
    s["phase_origin_count"] = 123
    s["phase_origin_timestamp_us"] = 900000
    s["target_relative_flap_phase_rad"] = .5
    s["target_flap_frequency_hz"] = 3.
    s["target_attitude_measurement_us"] = times
    s["target_rate_measurement_us"] = times
    s.to_parquet(tmp_path / "samples_train.parquet", index=False)
    pd.DataFrame([dict(flight_id="flight", sampling_group="main_50hz", history_us=250000,
                      horizon_us=100000, start_sample=0, origin_sample=6, end_sample=11)]).to_parquet(
                          tmp_path / "windows_train.parquet", index=False)
    for name in ["actuator_motors", "actuator_servos"]:
        raw = native(times - 100, np.full(n, .1), "control[0]")
        raw["flight_id"] = "flight"
        raw["control[1]"] = .2
        raw["control[2]"] = .3
        raw.to_parquet(tmp_path / f"native_{name}_train.parquet", index=False)
    (tmp_path / "dataset_manifest.json").write_text(json.dumps({"config": {"control_age_us": 100000}}))
    before = VelocityRateDataset(tmp_path, history_us=250000, horizon_us=100000, verify_hashes=False)[0]
    # Poison every future true state, phase and frequency, including historical-schema
    # columns stored on future rows. None may be copied into the recurrent input.
    poison = STATE + TARGET + PHASE + ["relative_flap_phase_unwrapped_rad",
                                     "target_relative_flap_phase_rad", "target_flap_frequency_hz"]
    s.loc[7:, poison] = 1234567.
    s.to_parquet(tmp_path / "samples_train.parquet", index=False)
    after = VelocityRateDataset(tmp_path, history_us=250000, horizon_us=100000, verify_hashes=False)[0]

    def equal(a, b):
        if isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                equal(a[key], b[key])
        else:
            np.testing.assert_equal(a, b)
    equal(before["inputs"], after["inputs"])
    assert not np.array_equal(before["targets"]["state"], after["targets"]["state"])


def test_frozen_split_preserves_old_test_dates_and_rejects_reassignment():
    cfg = config()
    cfg["split_dates"] = {p: [str(d) for d in dates] for p, dates in cfg["split_dates"].items()}
    cfg["frozen_test_dates"] = [str(d) for d in cfg["frozen_test_dates"]]
    source = Path(cfg["source_root"])
    if not source.exists():
        pytest.skip("local reviewed inventory unavailable")
    f, _, manifest = prepare_split(cfg, ROOT, source)
    assert len(f) == 194
    assert f.groupby("flight_date").split.nunique().max() == 1
    assert f.groupby("sha256").split.nunique().max() == 1
    assert set(f.loc[f.flight_date.isin(["2026-09-08", "2026-09-19"]), "split"]) == {"test"}
    cfg["split_dates"]["test"].remove("2026-09-19")
    cfg["split_dates"]["train"].append("2026-09-19")
    with pytest.raises(ValueError, match="reserved"):
        prepare_split(cfg, ROOT, source)


def test_kinematic_diagnostic_uses_actual_dt_and_body_quaternion_order():
    from system_identification.data.velocity_rate_dataset import TARGET
    from system_identification.data.velocity_rate_kinematics import endpoint_residual
    times = np.array([0, 20000, 45000, 60000])
    state = np.zeros((4, 13))
    state[:, 0] = times*1e-6*2
    state[:, 3] = 2
    state[:, 6] = np.cos(times*1e-6/2)
    state[:, 9] = np.sin(times*1e-6/2)
    state[:, 12] = 1
    frame = pd.DataFrame(state, columns=TARGET)
    frame["state_measurement_us"] = times
    w = pd.DataFrame([dict(origin_sample=0, future_count=3)])
    before = frame.copy(deep=True)
    position, angle = endpoint_residual(frame, w)
    np.testing.assert_allclose(position, 0, atol=1e-12)
    np.testing.assert_allclose(angle, 0, atol=1e-5)
    pd.testing.assert_frame_equal(frame, before)


def materialized_root():
    value = os.environ.get("VELOCITY_RATE_DATASET_QA")
    if not value:
        pytest.skip("set VELOCITY_RATE_DATASET_QA for explicit materialized split QA")
    return Path(value)


def test_materialized_full_future_event_tapes_are_valid():
    root = materialized_root()
    for part in ("train", "validation", "test"):
        windows = pd.read_parquet(root / f"windows_{part}.parquet")
        for topic, fields in (("actuator_motors", ["control[0]"]),
                              ("actuator_servos", ["control[0]", "control[1]", "control[2]"])):
            native = pd.read_parquet(root / f"native_{topic}_{part}.parquet")
            groups = dict(tuple(native.groupby("flight_id", sort=False)))
            for fid, w in windows.groupby("flight_id", sort=False):
                raw = groups[fid]
                timestamp = raw.timestamp_us.to_numpy()
                # Union of every origin-to-final-target half-open control interval.
                coverage = np.searchsorted(np.sort(w.initial_measurement_us), timestamp, side="right")
                coverage -= np.searchsorted(np.sort(w.last_target_measurement_us), timestamp, side="right")
                values = raw.loc[coverage > 0, fields].to_numpy()
                assert np.isfinite(values).all()
                assert (np.abs(values) <= 1).all()
                if topic == "actuator_motors":
                    assert (values >= 0).all()


def test_materialized_normalization_is_exactly_train_only():
    root = materialized_root()
    train = pd.read_parquet(root / "samples_train.parquet")
    normalization = json.loads((root / "normalization.json").read_text())
    assert normalization["partition"] == "train"
    for col, stats in normalization["columns"].items():
        mask = train.valid_core & np.isfinite(train[col])
        if col.startswith("relative_flap_phase_"):
            mask &= train.valid_encoder
        if col == "flap_frequency_hz":
            mask &= train.valid_frequency
        values = train.loc[mask, col].to_numpy(dtype=float)
        assert stats["count"] == len(values)
        np.testing.assert_allclose(stats["mean"], values.mean(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(stats["std"], values.std(ddof=1), rtol=1e-12, atol=1e-12)
