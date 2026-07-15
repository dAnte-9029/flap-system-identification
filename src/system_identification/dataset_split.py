"""Compatibility wrapper for :mod:`system_identification.data.splits`.

New code should import split utilities from the canonical module. Private helper
re-exports are retained because existing repository scripts import them.
"""

from system_identification.data.splits import (
    VALID_ROW_COLUMNS,
    _apply_altitude_window_trim,
    _cycle_to_block_map,
    _empty_blocks_frame,
    _load_log_records_from_manifest,
    _require_columns,
    _split_block_counts,
    _split_rows_for_log,
    _apply_purge_mask,
    _valid_row_mask,
    assign_cycle_block_splits,
    assign_log_splits,
    build_train_purge_intervals,
    extract_cycle_blocks,
    materialize_cycle_block_split,
    materialize_log_split,
)

__all__ = [
    "VALID_ROW_COLUMNS",
    "assign_cycle_block_splits",
    "assign_log_splits",
    "build_train_purge_intervals",
    "extract_cycle_blocks",
    "materialize_cycle_block_split",
    "materialize_log_split",
]
