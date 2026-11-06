"""Read-only PACLD view over frozen NativeWindows; no config.latent dependency."""
from dataclasses import replace
import hashlib
import numpy as np
import torch
from .pahbd import NativeWindows


class PACLDWindows:
    def __init__(self, dataset, indices=None):
        if dataset.split not in ('train', 'validation'):
            raise ValueError('sealed split forbidden')
        self.dataset = dataset
        frame = dataset.windows
        self.indices = list(frame.index[frame.shared_periodic_eligible] if indices is None else indices)
        if not self.indices or not frame.loc[self.indices, 'shared_periodic_eligible'].all():
            raise ValueError('nonempty Hall-eligible index required')
        self.index = frame.loc[self.indices].reset_index(drop=True).copy()
        if set(self.index.split) != {dataset.split}:
            raise ValueError('mixed split')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        window = self.dataset[self.indices[index]]
        if not window.metadata['shared_periodic_eligible'] or not window.metadata['periodic_enabled']:
            raise ValueError('no eligible causal Hall reference')
        return window  # PACLD.encode calls initial_state; never mutate window.initial.h.

    def identity(self):
        columns = ['window_id', 'flight_id', 'horizon_us', 'sampling_group']
        return dict(split=self.dataset.split, windows=len(self),
                    manifest_sha256=self.dataset.manifest_sha256,
                    index_sha256=hashlib.sha256(self.index[columns].to_csv(index=False).encode()).hexdigest(),
                    phase_contract=self.dataset.phase_contract,
                    phase_policy=self.dataset.phase_policy)


def open_windows(config, split, device='cpu', dtype=torch.float64, indices=None):
    if split not in ('train', 'validation'):
        raise ValueError('sealed test is not an allowed split')
    data = config['data']
    if data['history_us'] != 500000 or sorted(data['horizon_us']) != [200000, 500000, 1000000, 2000000]:
        raise ValueError('PACLD v1 requires frozen 0.5s history and four horizons')
    if sorted(data['sampling_groups']) != ['main_50hz', 'sparse_10hz'] or data['phase_policy'] != 'shared_or_average':
        raise ValueError('native sampling groups and shared Hall contract required')
    dataset = NativeWindows(**data, split=split, device=device, dtype=dtype)
    # Phase v1 config overrides only the descriptive scope; numeric contract is frozen.
    expected = dict(config['phase_contract']); actual = dict(dataset.phase_contract)
    expected.pop('frozen_contract_override', None); actual.pop('frozen_contract_override', None)
    if expected != actual:
        raise ValueError('configured Hall contract differs from frozen reader')
    return PACLDWindows(dataset, indices)


def fixed_validation_indices(index, seed=1701, per_group=1):
    if per_group < 1 or set(index.split) != {'validation'}:
        raise ValueError('positive validation selection required')
    rng = np.random.default_rng(seed)
    return [int(i) for _, g in index.groupby(['flight_id', 'horizon_us', 'sampling_group'], sort=True)
            for i in rng.choice(g.index, min(per_group, len(g)), replace=False)]


def subset(windows, positions):
    return PACLDWindows(windows.dataset, [windows.indices[i] for i in positions])


def move_window(window, device, dtype=torch.float64):
    """Move existing tensors only; preserve all metadata and native event clocks."""
    def move(v):
        return v.to(device=device, dtype=dtype) if torch.is_tensor(v) else v
    initial = replace(window.initial, **{k: move(v) for k, v in vars(window.initial).items()})
    def observation(e):
        return replace(e, value=move(e.value), mask=move(e.mask), variance=move(e.variance), quality=move(e.quality))
    return replace(window, initial=initial, history=[observation(e) for e in window.history],
                   targets=[observation(e) for e in window.targets],
                   controls=replace(window.controls, values=move(window.controls.values)))
