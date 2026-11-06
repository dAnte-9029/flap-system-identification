"""Read-only PACLD view over frozen NativeWindows; no config.latent dependency."""
from dataclasses import replace
from collections import OrderedDict
import hashlib
import numpy as np
import torch
from .pahbd import NativeWindows


class CachedNativeWindows(NativeWindows):
    """Bounded read-only per-instance LRU. Keys include split and flight identity.

    NativeWindows owns phase tables/index already; this cache holds only the
    lazily loaded flight tables, samples and encoder counts, never windows.
    """
    def __init__(self, *args, cache_flights=8, cache_bytes=268435456, **kwargs):
        if cache_flights < 0 or cache_bytes < 0:
            raise ValueError('nonnegative cache bounds required')
        self.cache_flights = cache_flights
        self.cache_bytes = cache_bytes
        self._flights = OrderedDict()
        self._cache_stats = dict(hits=0, misses=0, evictions=0, parquet_reads=0,
                                 retained_bytes=0, peak_retained_bytes=0, active_bytes=0)
        self._active_encoder = None
        super().__init__(*args, **kwargs)

    def _load(self, fid):
        key = (self.split, fid)
        if key in self._flights:
            self._cache_stats['hits'] += 1
            payload = self._flights.pop(key)
            self._flights[key] = payload
        else:
            self._cache_stats['misses'] += 1
            self._cached_id = None  # force a read after eviction, including oversized entries
            super()._load(fid)
            encoder = super()._encoder_counts(fid)
            self._cache_stats['parquet_reads'] += 8  # six topics + samples + encoder
            frames = [v for k, v in self.tables.items() if k != 'wing_phase']+[self.samples, encoder]
            size = sum(int(frame.memory_usage(index=True, deep=True).sum()) for frame in frames)
            payload = (self.tables, self.samples, encoder, size)
            if self.cache_flights and size <= self.cache_bytes:
                while self._flights and (len(self._flights) >= self.cache_flights or
                                         self._cache_stats['retained_bytes']+size > self.cache_bytes):
                    _, old = self._flights.popitem(last=False)
                    self._cache_stats['retained_bytes'] -= old[3]
                    self._cache_stats['evictions'] += 1
                self._flights[key] = payload
                self._cache_stats['retained_bytes'] += size
                self._cache_stats['peak_retained_bytes'] = max(self._cache_stats['peak_retained_bytes'],
                                                              self._cache_stats['retained_bytes'])
        self.tables, self.samples, self._active_encoder, size = payload
        self._cache_stats['active_bytes'] = size
        self._cached_id = fid

    def _encoder_counts(self, fid):
        if fid != self._cached_id or self._active_encoder is None:
            raise ValueError('encoder requested outside active flight')
        return self._active_encoder

    def cache_stats(self):
        return dict(self._cache_stats, entries=len(self._flights), split=self.split,
                    max_flights=self.cache_flights, max_bytes=self.cache_bytes)


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
    cache = config.get('data_cache', {})
    dataset = CachedNativeWindows(**data, split=split, device=device, dtype=dtype,
                                  cache_flights=cache.get('max_flights', 8),
                                  cache_bytes=cache.get('max_bytes', 268435456))
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
