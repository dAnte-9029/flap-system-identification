"""Canonical representation of logged, microsecond-resolution ZOH commands.

Each row is [motor, left, right, rudder, duration_seconds]. Leading axes
identify independent state intervals: merging NEVER crosses those axes.
This is representation normalization, not actuator dynamics or resampling.
"""
import torch


CANONICAL_CONTROL_EVENTS = "exact_zoh_merge_us_v1"
LEGACY_CONTROL_EVENTS = "legacy_raw_v0"
TIME_QUANTIZATION_ATOL_US = 0.05  # float32 storage tolerance, NOT a merge tolerance


def canonicalize_control_events(events, lengths, step_dt=None):
    """Return compact, zero-padded events and lengths without mutating inputs.

    Compare all four stored command values exactly. Remove zero-duration rows,
    merge adjacent equal commands, and sum durations as integer microseconds.
    Arbitrary padding (including NaN) is ignored. An empty/zero-duration state
    interval becomes one zero token, matching masked rollout padding.

    ``step_dt``, when supplied, must agree with the total interval duration on
    the same integer clock. Sub-microsecond inputs fail rather than silently
    changing their timing. Model weights remain differentiable; the discrete
    segmentation/timestamp decisions are not a differentiable control API.
    """
    if events.ndim < 2 or events.shape[-1] != 5 or events.shape[-2] < 1:
        raise ValueError("events must have shape [..., E>=1, 5]")
    if events.dtype not in (torch.float32, torch.float64):
        raise ValueError("events must use float32 or float64")
    if lengths.shape != events.shape[:-2] or lengths.dtype != torch.int64:
        raise ValueError("lengths must be int64 with the leading event shape")
    if lengths.device != events.device or lengths.numel() == 0:
        raise ValueError("nonempty lengths must share the event device")
    width = events.shape[-2]
    ev = events.reshape(-1, width, 5)
    le = lengths.reshape(-1)
    if ((le < 0) | (le > width)).any():
        raise ValueError("event length outside [0, E]")
    index = torch.arange(width, device=events.device).expand(len(ev), -1)
    used = index < le[:, None]
    safe = torch.where(used[..., None], ev, 0)
    if not torch.isfinite(safe).all() or (safe[..., 4] < 0).any():
        raise ValueError("valid events must be finite with nonnegative durations")
    ticks = safe[..., 4].double() * 1_000_000
    rounded = ticks.round()
    if ((ticks - rounded).abs() > TIME_QUANTIZATION_ATOL_US).any():
        raise ValueError("event durations must lie on the integer microsecond clock")
    # Reject positive intervals that would otherwise disappear on rounding.
    if ((safe[..., 4] > 0) & (rounded == 0)).any():
        raise ValueError("positive sub-microsecond duration is not representable")
    durations = rounded.to(torch.int64)
    if step_dt is not None:
        if step_dt.shape != lengths.shape or step_dt.device != events.device:
            raise ValueError("step_dt must match lengths shape and device")
        dt = step_dt.reshape(-1).double() * 1_000_000
        if (not torch.isfinite(dt).all() or (dt < 0).any()
                or ((dt - dt.round()).abs() > TIME_QUANTIZATION_ATOL_US).any()
                or not torch.equal(dt.round().long(), durations.sum(-1))):
            raise ValueError("event durations do not exactly cover the state interval")

    positive = used & (durations > 0)
    last = torch.where(positive, index, -1).cummax(-1).values
    previous = torch.cat((last.new_full((len(ev), 1), -1), last[:, :-1]), -1)
    previous_u = safe[..., :4].gather(1, previous.clamp_min(0)[..., None].expand(-1, -1, 4))
    start = positive & ((previous < 0) | (safe[..., :4] != previous_u).any(-1))
    group = (start.long().cumsum(-1) - 1).clamp_min(0)
    new_lengths = start.sum(-1).clamp_min(1)
    new_width = int(new_lengths.max().item())
    commands = events.new_zeros((len(ev), new_width, 4))
    row, col = start.nonzero(as_tuple=True)
    commands[row, group[row, col]] = safe[row, col, :4]
    # Give signed zero one canonical bit representation as well.
    commands = torch.where(commands == 0, 0, commands)
    duration_sum = durations.new_zeros((len(ev), new_width))
    duration_sum.scatter_add_(1, group, durations)
    seconds = (duration_sum.double() * 1e-6).to(events.dtype)
    result = torch.cat((commands, seconds[..., None]), -1)
    return (result.reshape(*lengths.shape, new_width, 5),
            new_lengths.reshape(lengths.shape))


def canonicalize_control_events_numpy(events, lengths, step_dt=None):
    """CPU data-packing adapter using the SAME implementation as inference."""
    ev = torch.as_tensor(events)
    le = torch.as_tensor(lengths)
    dt = None if step_dt is None else torch.as_tensor(step_dt)
    out, length = canonicalize_control_events(ev, le, dt)
    return out.numpy(), length.numpy()
