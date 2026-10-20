"""Horizon-specific loss coverage, padding, and long recurrent gradients."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_velocity_rate_horizons import crop_horizon
from train_velocity_rate_horizons import CACHE, OUT, ROOT
from train_velocity_rate_learning import window_loss
from test_velocity_rate_long_horizon import inputs
from system_identification.models.velocity_rate_gru import VelocityRateGRU
from system_identification.data.velocity_rate_dataset import VelocityRateDataset


def test_crop_uses_registered_variable_lengths_and_removes_later_controls():
    data = dict(initial=np.zeros((2, 13)), target=np.arange(2*51*13).reshape(2, 51, 13),
                dt=np.full((2, 51), .02), future_mask=np.ones((2, 51), bool),
                events=np.ones((2, 51, 3, 5)), event_length=np.full((2, 51), 3))
    cropped = crop_horizon(data, np.array([25, 26]))
    assert cropped["target"].shape == (2, 26, 13)
    assert cropped["future_mask"].sum(1).tolist() == [25, 26]
    assert cropped["dt"][0, 25] == 0 and cropped["dt"][1, 25] == .02
    assert not cropped["events"][0, 25].any()
    assert cropped["event_length"][0, 25] == 1
    np.testing.assert_array_equal(cropped["target"][0, 25], data["target"][0, 24])
    assert data["events"][0, 25].all()  # Input pack remains unchanged.


@pytest.mark.parametrize("steps", [25, 50])
def test_all_steps_contribute_and_masked_padding_does_not(steps):
    target = torch.zeros(2, steps+1, 13)
    target[..., 6] = 1
    pred = target.clone()
    pred[:, :steps, 3] = torch.arange(1, steps+1)
    pred[:, steps, 3] = 1e6
    mask = torch.arange(steps+1)[None].expand(2, -1) < steps
    loss = window_loss(pred, target, mask, torch.ones(4), torch.ones(4))
    torch.testing.assert_close(loss, torch.full((2,), torch.arange(1, steps+1).float().square().mean()))


@pytest.mark.parametrize("control", [False, True])
def test_one_second_terminal_loss_backpropagates_through_first_state(control):
    torch.set_num_threads(2)
    torch.manual_seed(17)
    m = VelocityRateGRU(dict(mean=[0]*11, scale=[1]*11, output_scale=[.1]*6), control=control).double()
    torch.nn.init.normal_(m.head.weight, std=.01)
    x = inputs()
    for key in ("dt", "events", "event_length"):
        x[key] = x[key][:, :50]
    x["initial"].requires_grad_()
    pred = m(x)
    loss = pred[:, -1, :6].square().mean() + pred[:, -1, 10:].square().mean()
    loss.backward()
    assert x["initial"].grad is not None and x["initial"].grad[:, 3:6].abs().sum() > 0
    assert m.history_gru.weight_ih_l0.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


@pytest.mark.parametrize("split", ["train", "validation"])
def test_new_pack_against_authoritative_loader_and_registered_endpoints(split):
    import pandas as pd
    if not (CACHE / f"{split}.npz").exists():
        pytest.skip("new experiment pack is not materialized")
    archive = np.load(CACHE / f"{split}.npz")
    data = {key: archive[key] for key in archive.files}
    ds = VelocityRateDataset(ROOT / "dataset/velocity_rate_multistep_v1", split=split,
                             horizon_us=1000000, verify_hashes=False)
    meta = pd.read_parquet(OUT / f"{split}_windows.parquet")
    assert len(ds) == len(meta)
    for i in sorted({0, len(ds)//2, len(ds)-1, int(np.argmax(meta.count_1000ms))}):
        sample = ds[i]
        inp = sample["inputs"]
        h, k = len(inp["history"]), len(inp["future_dt_s"])
        np.testing.assert_allclose(data["history"][i, :h, 3:], inp["history"][:, 3:], rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(data["history_mask"][i, :h], inp["history_mask"])
        np.testing.assert_allclose(data["dt"][i, :k], inp["future_dt_s"], atol=1e-8)
        expected = sample["targets"]["state"].copy()
        expected[:, :3] -= inp["initial_state"][:3]
        np.testing.assert_allclose(data["target"][i, :k], expected, rtol=1e-5, atol=3e-5)
        tape = inp["future_control"]
        for j in range(k):
            count = data["event_length"][i, j]
            events = data["events"][i, j, :count]
            np.testing.assert_allclose(events[:, 4].sum(), data["dt"][i, j], atol=1e-8)
            np.testing.assert_allclose(events[0, :4], tape["step_control"][j], atol=1e-7)
            controls = tape["event_control"][tape["event_step_index"] == j]
            assert count in (len(controls), len(controls)+1)
            if len(controls):
                np.testing.assert_allclose(events[-len(controls):, :4], controls, atol=1e-7)
    for horizon in (500, 1000):
        cropped = crop_horizon(data, meta[f"count_{horizon}ms"].to_numpy())
        assert np.max(abs(cropped["dt"].sum(1)-horizon/1000)) < .012001
        assert (cropped["future_mask"].sum(1) == meta[f"count_{horizon}ms"].to_numpy()).all()
