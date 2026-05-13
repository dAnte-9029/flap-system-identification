#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from system_identification.training import run_baseline_comparison


@dataclass(frozen=True)
class ScreenConfig:
    config_id: str
    stage: str
    recipe_name: str
    hidden_sizes: tuple[int, ...]
    sequence_history_size: int
    max_epochs: int
    early_stopping_patience: int
    learning_rate: float
    weight_decay: float
    dropout: float
    extra_args: dict[str, int | float | str | None]


def classify_candidate(
    *,
    candidate_rmse: float,
    reference_rmse: float,
    candidate_r2: float,
    reference_r2: float,
    hard_target_improvements: int,
    worst_regime_rmse_improvement: float,
) -> str:
    if candidate_rmse <= 0.97 * reference_rmse:
        return "promote"
    if candidate_r2 >= reference_r2 + 0.02:
        return "promote"
    if candidate_rmse <= 1.03 * reference_rmse and hard_target_improvements >= 2:
        return "promote"
    if candidate_rmse <= 1.03 * reference_rmse and worst_regime_rmse_improvement >= 0.05:
        return "promote"
    if candidate_rmse <= 1.03 * reference_rmse:
        return "ablation"
    if candidate_rmse > 1.05 * reference_rmse:
        return "reject"
    return "ablation"


def _config(
    *,
    config_id: str,
    stage: str,
    recipe_name: str,
    hidden_sizes: tuple[int, ...],
    sequence_history_size: int,
    max_epochs: int,
    early_stopping_patience: int,
    dropout: float,
    extra_args: dict[str, int | float | str | None] | None = None,
) -> ScreenConfig:
    return ScreenConfig(
        config_id=config_id,
        stage=stage,
        recipe_name=recipe_name,
        hidden_sizes=hidden_sizes,
        sequence_history_size=sequence_history_size,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        learning_rate=1e-3,
        weight_decay=1e-5,
        dropout=dropout,
        extra_args=extra_args or {},
    )


def _quick_configs() -> list[ScreenConfig]:
    return [
        _config(
            config_id="quick_mlp_h128x2",
            stage="quick",
            recipe_name="mlp_paper_no_accel_v2",
            hidden_sizes=(128, 128),
            sequence_history_size=64,
            max_epochs=12,
            early_stopping_patience=4,
            dropout=0.05,
        ),
        _config(
            config_id="quick_gru_h128x2_hist64",
            stage="quick",
            recipe_name="causal_gru_paper_no_accel_v2_phase_actuator_airdata",
            hidden_sizes=(128, 128),
            sequence_history_size=64,
            max_epochs=12,
            early_stopping_patience=4,
            dropout=0.05,
        ),
        _config(
            config_id="quick_lstm_h128x2_hist64",
            stage="quick",
            recipe_name="causal_lstm_paper_no_accel_v2_phase_actuator_airdata",
            hidden_sizes=(128, 128),
            sequence_history_size=64,
            max_epochs=12,
            early_stopping_patience=4,
            dropout=0.05,
        ),
        _config(
            config_id="quick_tcn_c128_b4_k3_hist64",
            stage="quick",
            recipe_name="causal_tcn_paper_no_accel_v2_phase_actuator_airdata",
            hidden_sizes=(128, 128),
            sequence_history_size=64,
            max_epochs=12,
            early_stopping_patience=4,
            dropout=0.05,
            extra_args={"tcn_channels": 128, "tcn_num_blocks": 4, "tcn_kernel_size": 3},
        ),
        _config(
            config_id="quick_transformer_d64_l1_h4_hist64",
            stage="quick",
            recipe_name="causal_transformer_paper_no_accel_v2_phase_actuator_airdata",
            hidden_sizes=(64, 128),
            sequence_history_size=64,
            max_epochs=12,
            early_stopping_patience=4,
            dropout=0.05,
            extra_args={
                "transformer_d_model": 64,
                "transformer_num_layers": 1,
                "transformer_num_heads": 4,
                "transformer_dim_feedforward": 128,
            },
        ),
        _config(
            config_id="quick_tcn_gru_c128_b3_k3_h128_hist64",
            stage="quick",
            recipe_name="causal_tcn_gru_paper_no_accel_v2_phase_actuator_airdata",
            hidden_sizes=(128, 128),
            sequence_history_size=64,
            max_epochs=12,
            early_stopping_patience=4,
            dropout=0.05,
            extra_args={"tcn_channels": 128, "tcn_num_blocks": 3, "tcn_kernel_size": 3},
        ),
    ]


def _sweep_configs() -> list[ScreenConfig]:
    configs: list[ScreenConfig] = [
        _config(
            config_id="sweep_gru_h128_hist64",
            stage="sweep",
            recipe_name="causal_gru_paper_no_accel_v2_phase_actuator_airdata",
            hidden_sizes=(128, 128),
            sequence_history_size=64,
            max_epochs=20,
            early_stopping_patience=5,
            dropout=0.05,
        )
    ]
    for history in (32, 64, 128):
        for hidden in (64, 128):
            configs.append(
                _config(
                    config_id=f"sweep_lstm_h{hidden}_l1_hist{history}",
                    stage="sweep",
                    recipe_name="causal_lstm_paper_no_accel_v2_phase_actuator_airdata",
                    hidden_sizes=(hidden, hidden),
                    sequence_history_size=history,
                    max_epochs=20,
                    early_stopping_patience=5,
                    dropout=0.05,
                    extra_args={"gru_num_layers": 1},
                )
            )
    for history, channels, blocks, kernel in (
        (32, 64, 3, 3),
        (32, 128, 3, 3),
        (64, 64, 4, 3),
        (64, 128, 4, 3),
        (64, 128, 5, 3),
        (64, 128, 4, 5),
        (128, 64, 4, 3),
        (128, 128, 4, 3),
    ):
        configs.append(
            _config(
                config_id=f"sweep_tcn_c{channels}_b{blocks}_k{kernel}_hist{history}",
                stage="sweep",
                recipe_name="causal_tcn_paper_no_accel_v2_phase_actuator_airdata",
                hidden_sizes=(channels, 128),
                sequence_history_size=history,
                max_epochs=20,
                early_stopping_patience=5,
                dropout=0.05,
                extra_args={"tcn_channels": channels, "tcn_num_blocks": blocks, "tcn_kernel_size": kernel},
            )
        )
    for history, d_model, layers, heads in (
        (64, 32, 1, 2),
        (64, 64, 1, 4),
        (64, 64, 2, 4),
        (128, 32, 1, 2),
        (128, 64, 1, 4),
        (128, 64, 2, 4),
    ):
        configs.append(
            _config(
                config_id=f"sweep_transformer_d{d_model}_l{layers}_h{heads}_hist{history}",
                stage="sweep",
                recipe_name="causal_transformer_paper_no_accel_v2_phase_actuator_airdata",
                hidden_sizes=(d_model, 128),
                sequence_history_size=history,
                max_epochs=20,
                early_stopping_patience=5,
                dropout=0.05,
                extra_args={
                    "transformer_d_model": d_model,
                    "transformer_num_layers": layers,
                    "transformer_num_heads": heads,
                    "transformer_dim_feedforward": 2 * d_model,
                },
            )
        )
    for history, channels, blocks, hidden in (
        (64, 64, 2, 64),
        (64, 128, 3, 128),
        (128, 64, 2, 64),
        (128, 128, 3, 128),
    ):
        configs.append(
            _config(
                config_id=f"sweep_tcn_gru_c{channels}_b{blocks}_h{hidden}_hist{history}",
                stage="sweep",
                recipe_name="causal_tcn_gru_paper_no_accel_v2_phase_actuator_airdata",
                hidden_sizes=(hidden, hidden),
                sequence_history_size=history,
                max_epochs=20,
                early_stopping_patience=5,
                dropout=0.05,
                extra_args={"tcn_channels": channels, "tcn_num_blocks": blocks, "tcn_kernel_size": 3},
            )
        )
    return configs


def _final_configs() -> list[ScreenConfig]:
    return [
        _config(
            config_id="final_mlp_h128x2",
            stage="final",
            recipe_name="mlp_paper_no_accel_v2",
            hidden_sizes=(128, 128),
            sequence_history_size=64,
            max_epochs=50,
            early_stopping_patience=8,
            dropout=0.0,
        ),
        _config(
            config_id="final_gru_h128_hist64",
            stage="final",
            recipe_name="causal_gru_paper_no_accel_v2_phase_actuator_airdata",
            hidden_sizes=(128, 128),
            sequence_history_size=64,
            max_epochs=50,
            early_stopping_patience=8,
            dropout=0.0,
        ),
        _config(
            config_id="final_transformer_d64_l2_h4_hist128",
            stage="final",
            recipe_name="causal_transformer_paper_no_accel_v2_phase_actuator_airdata",
            hidden_sizes=(64, 128),
            sequence_history_size=128,
            max_epochs=50,
            early_stopping_patience=8,
            dropout=0.0,
            extra_args={
                "transformer_d_model": 64,
                "transformer_num_layers": 2,
                "transformer_num_heads": 4,
                "transformer_dim_feedforward": 128,
            },
        ),
        _config(
            config_id="final_tcn_c128_b4_k3_hist128",
            stage="final",
            recipe_name="causal_tcn_paper_no_accel_v2_phase_actuator_airdata",
            hidden_sizes=(128, 128),
            sequence_history_size=128,
            max_epochs=50,
            early_stopping_patience=8,
            dropout=0.0,
            extra_args={"tcn_channels": 128, "tcn_num_blocks": 4, "tcn_kernel_size": 3},
        ),
        _config(
            config_id="final_tcn_gru_c64_b2_h64_hist128",
            stage="final",
            recipe_name="causal_tcn_gru_paper_no_accel_v2_phase_actuator_airdata",
            hidden_sizes=(64, 64),
            sequence_history_size=128,
            max_epochs=50,
            early_stopping_patience=8,
            dropout=0.0,
            extra_args={"tcn_channels": 64, "tcn_num_blocks": 2, "tcn_kernel_size": 3},
        ),
    ]


def _tcn_gru_focused_configs(*, final: bool = False) -> list[ScreenConfig]:
    stage = "tcn_gru_focused_final" if final else "tcn_gru_focused"
    max_epochs = 50 if final else 20
    patience = 8 if final else 5
    specs = [
        (128, 64, 2, 3, 64),
        (128, 96, 2, 3, 64),
        (128, 128, 2, 3, 64),
        (128, 64, 3, 3, 64),
        (128, 64, 4, 3, 64),
        (128, 64, 2, 5, 64),
        (96, 64, 2, 3, 64),
        (160, 64, 2, 3, 64),
        (128, 64, 2, 3, 96),
        (128, 64, 2, 3, 128),
        (160, 96, 3, 3, 96),
        (160, 128, 4, 3, 128),
    ]
    configs: list[ScreenConfig] = []
    for history, channels, blocks, kernel, gru_hidden in specs:
        configs.append(
            _config(
                config_id=f"{stage}_hist{history}_c{channels}_b{blocks}_k{kernel}_gru{gru_hidden}",
                stage=stage,
                recipe_name="causal_tcn_gru_paper_no_accel_v2_phase_actuator_airdata",
                hidden_sizes=(gru_hidden, gru_hidden),
                sequence_history_size=history,
                max_epochs=max_epochs,
                early_stopping_patience=patience,
                dropout=0.0,
                extra_args={
                    "tcn_channels": channels,
                    "tcn_num_blocks": blocks,
                    "tcn_kernel_size": kernel,
                },
            )
        )
    return configs


def _transformer_focused_configs(*, final: bool = False) -> list[ScreenConfig]:
    stage = "transformer_focused_final" if final else "transformer_focused"
    max_epochs = 50 if final else 20
    patience = 8 if final else 5
    specs = [
        (96, 64, 2, 4, 0.0),
        (128, 64, 2, 4, 0.0),
        (160, 64, 2, 4, 0.0),
        (192, 64, 2, 4, 0.0),
        (128, 64, 1, 4, 0.0),
        (128, 64, 3, 4, 0.0),
        (128, 96, 2, 4, 0.0),
        (160, 96, 2, 4, 0.0),
        (128, 128, 2, 4, 0.0),
        (128, 64, 2, 2, 0.0),
        (128, 64, 2, 8, 0.0),
        (128, 64, 2, 4, 0.05),
    ]
    configs: list[ScreenConfig] = []
    for history, d_model, layers, heads, dropout in specs:
        dropout_tag = "do0" if dropout == 0.0 else f"do{int(dropout * 1000):03d}"
        configs.append(
            _config(
                config_id=f"{stage}_hist{history}_d{d_model}_l{layers}_h{heads}_{dropout_tag}",
                stage=stage,
                recipe_name="causal_transformer_paper_no_accel_v2_phase_actuator_airdata",
                hidden_sizes=(d_model, 128),
                sequence_history_size=history,
                max_epochs=max_epochs,
                early_stopping_patience=patience,
                dropout=dropout,
                extra_args={
                    "transformer_d_model": d_model,
                    "transformer_num_layers": layers,
                    "transformer_num_heads": heads,
                    "transformer_dim_feedforward": 2 * d_model,
                },
            )
        )
    return configs


def _phase_harmonic_configs(*, final: bool = False) -> list[ScreenConfig]:
    stage = "phase_harmonic_final" if final else "phase_harmonic"
    max_epochs = 50 if final else 20
    patience = 8 if final else 5
    specs = [
        ("no_phase", "causal_transformer_paper_no_accel_v2_no_phase_airdata"),
        ("raw_phase", "causal_transformer_paper_no_accel_v2_raw_phase_airdata"),
        ("sin_cos", "causal_transformer_paper_no_accel_v2_phase_actuator_airdata"),
        ("harmonic3", "causal_transformer_paper_no_accel_v2_phase_harmonic_airdata"),
    ]
    configs: list[ScreenConfig] = []
    for label, recipe_name in specs:
        configs.append(
            _config(
                config_id=f"{stage}_{label}",
                stage=stage,
                recipe_name=recipe_name,
                hidden_sizes=(64, 128),
                sequence_history_size=128,
                max_epochs=max_epochs,
                early_stopping_patience=patience,
                dropout=0.05,
                extra_args={
                    "transformer_d_model": 64,
                    "transformer_num_layers": 2,
                    "transformer_num_heads": 4,
                    "transformer_dim_feedforward": 128,
                },
            )
        )
    return configs


def _phase_film_configs(*, final: bool = False) -> list[ScreenConfig]:
    stage = "phase_film_final" if final else "phase_film"
    max_epochs = 50 if final else 20
    patience = 8 if final else 5
    specs = [
        ("baseline", "causal_transformer_paper_no_accel_v2_phase_actuator_airdata"),
        ("head", "causal_transformer_head_film_paper_no_accel_v2_phase_actuator_airdata"),
        ("input", "causal_transformer_input_film_paper_no_accel_v2_phase_actuator_airdata"),
    ]
    configs: list[ScreenConfig] = []
    for label, recipe_name in specs:
        configs.append(
            _config(
                config_id=f"{stage}_{label}",
                stage=stage,
                recipe_name=recipe_name,
                hidden_sizes=(64, 128),
                sequence_history_size=128,
                max_epochs=max_epochs,
                early_stopping_patience=patience,
                dropout=0.05,
                extra_args={
                    "transformer_d_model": 64,
                    "transformer_num_layers": 2,
                    "transformer_num_heads": 4,
                    "transformer_dim_feedforward": 128,
                },
            )
        )
    return configs


def _history_length_configs() -> list[ScreenConfig]:
    histories = (1, 16, 32, 64, 128, 256)
    configs: list[ScreenConfig] = [
        _config(
            config_id="history_mlp_current",
            stage="history_length",
            recipe_name="mlp_paper_no_accel_v2",
            hidden_sizes=(128, 128),
            sequence_history_size=1,
            max_epochs=20,
            early_stopping_patience=5,
            dropout=0.05,
        )
    ]
    for history in histories:
        configs.extend(
            [
                _config(
                    config_id=f"history_gru_h128_hist{history}",
                    stage="history_length",
                    recipe_name="causal_gru_paper_no_accel_v2_phase_actuator_airdata",
                    hidden_sizes=(128, 128),
                    sequence_history_size=history,
                    max_epochs=20,
                    early_stopping_patience=5,
                    dropout=0.05,
                ),
                _config(
                    config_id=f"history_tcn_c128_b4_k3_hist{history}",
                    stage="history_length",
                    recipe_name="causal_tcn_paper_no_accel_v2_phase_actuator_airdata",
                    hidden_sizes=(128, 128),
                    sequence_history_size=history,
                    max_epochs=20,
                    early_stopping_patience=5,
                    dropout=0.05,
                    extra_args={"tcn_channels": 128, "tcn_num_blocks": 4, "tcn_kernel_size": 3},
                ),
                _config(
                    config_id=f"history_transformer_d64_l2_h4_hist{history}",
                    stage="history_length",
                    recipe_name="causal_transformer_paper_no_accel_v2_phase_actuator_airdata",
                    hidden_sizes=(64, 128),
                    sequence_history_size=history,
                    max_epochs=20,
                    early_stopping_patience=5,
                    dropout=0.05,
                    extra_args={
                        "transformer_d_model": 64,
                        "transformer_num_layers": 2,
                        "transformer_num_heads": 4,
                        "transformer_dim_feedforward": 128,
                    },
                ),
            ]
        )
    return configs


def _temporal_order_configs() -> list[ScreenConfig]:
    base_extra_args = {
        "transformer_d_model": 64,
        "transformer_num_layers": 2,
        "transformer_num_heads": 4,
        "transformer_dim_feedforward": 128,
    }
    return [
        _config(
            config_id="temporal_order_normal_hist128",
            stage="temporal_order",
            recipe_name="causal_transformer_paper_no_accel_v2_phase_actuator_airdata",
            hidden_sizes=(64, 128),
            sequence_history_size=128,
            max_epochs=20,
            early_stopping_patience=5,
            dropout=0.05,
            extra_args=base_extra_args,
        ),
        _config(
            config_id="temporal_order_no_pos_hist128",
            stage="temporal_order",
            recipe_name="causal_transformer_paper_no_accel_v2_phase_actuator_airdata",
            hidden_sizes=(64, 128),
            sequence_history_size=128,
            max_epochs=20,
            early_stopping_patience=5,
            dropout=0.05,
            extra_args={**base_extra_args, "transformer_use_positional_encoding": False},
        ),
        _config(
            config_id="temporal_order_h1",
            stage="temporal_order",
            recipe_name="causal_transformer_paper_no_accel_v2_phase_actuator_airdata",
            hidden_sizes=(64, 128),
            sequence_history_size=1,
            max_epochs=20,
            early_stopping_patience=5,
            dropout=0.05,
            extra_args=base_extra_args,
        ),
    ]


def build_screen_configs(stage: str) -> list[ScreenConfig]:
    resolved_stage = stage.lower()
    if resolved_stage == "quick":
        return _quick_configs()
    if resolved_stage == "sweep":
        return _sweep_configs()
    if resolved_stage == "final":
        return _final_configs()
    if resolved_stage == "tcn_gru_focused":
        return _tcn_gru_focused_configs(final=False)
    if resolved_stage == "tcn_gru_focused_final":
        return _tcn_gru_focused_configs(final=True)
    if resolved_stage == "transformer_focused":
        return _transformer_focused_configs(final=False)
    if resolved_stage == "transformer_focused_final":
        return _transformer_focused_configs(final=True)
    if resolved_stage == "phase_harmonic":
        return _phase_harmonic_configs(final=False)
    if resolved_stage == "phase_harmonic_final":
        return _phase_harmonic_configs(final=True)
    if resolved_stage == "phase_film":
        return _phase_film_configs(final=False)
    if resolved_stage == "phase_film_final":
        return _phase_film_configs(final=True)
    if resolved_stage == "history_length":
        return _history_length_configs()
    if resolved_stage == "temporal_order":
        return _temporal_order_configs()
    if resolved_stage == "all":
        return [
            *_quick_configs(),
            *_sweep_configs(),
            *_final_configs(),
            *_tcn_gru_focused_configs(final=False),
            *_tcn_gru_focused_configs(final=True),
            *_transformer_focused_configs(final=False),
            *_transformer_focused_configs(final=True),
            *_phase_harmonic_configs(final=False),
            *_phase_harmonic_configs(final=True),
            *_phase_film_configs(final=False),
            *_phase_film_configs(final=True),
            *_history_length_configs(),
            *_temporal_order_configs(),
        ]
    raise ValueError(f"Unknown stage: {stage}")


def _parse_hidden_sizes(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("hidden sizes must contain at least one integer")
    return values


def _config_row(config: ScreenConfig) -> dict[str, Any]:
    row = asdict(config)
    row["hidden_sizes"] = ",".join(str(value) for value in config.hidden_sizes)
    row.update({f"extra_{key}": value for key, value in config.extra_args.items()})
    row["extra_args"] = json.dumps(config.extra_args, sort_keys=True)
    return row


def _stage_sample_defaults(stage: str) -> tuple[int | None, int | None, int | None]:
    if stage == "quick":
        return 65536, 32768, 32768
    if stage in {
        "sweep",
        "tcn_gru_focused",
        "transformer_focused",
        "phase_harmonic",
        "phase_film",
        "history_length",
        "temporal_order",
    }:
        return 131072, 65536, 65536
    return None, None, None


def _run_config(
    *,
    config: ScreenConfig,
    split_root: str,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
    device: str,
    random_seed: int,
    use_amp: bool,
    max_train_samples: int | None,
    max_val_samples: int | None,
    max_test_samples: int | None,
    skip_test_eval: bool,
) -> dict[str, Any]:
    run_dir = output_dir / "runs" / config.config_id
    kwargs: dict[str, Any] = {
        "split_root": split_root,
        "output_dir": run_dir,
        "recipe_names": [config.recipe_name],
        "hidden_sizes": config.hidden_sizes,
        "dropout": config.dropout,
        "batch_size": batch_size,
        "max_epochs": config.max_epochs,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "early_stopping_patience": config.early_stopping_patience,
        "device": device,
        "random_seed": random_seed,
        "num_workers": num_workers,
        "use_amp": use_amp,
        "max_train_samples": max_train_samples,
        "max_val_samples": max_val_samples,
        "max_test_samples": max_test_samples,
        "skip_test_eval": skip_test_eval,
        "sequence_history_size": config.sequence_history_size,
    }
    kwargs.update(config.extra_args)
    outputs = run_baseline_comparison(**kwargs)
    summary = pd.read_csv(outputs["summary_csv_path"])
    if len(summary) != 1:
        raise RuntimeError(f"Expected one summary row for {config.config_id}, got {len(summary)}")
    row = summary.iloc[0].to_dict()
    return {**_config_row(config), **row, "run_output_dir": str(run_dir)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run staged screening for deployable temporal backbones")
    parser.add_argument("--split-root", required=True, help="Whole-log dataset split root")
    parser.add_argument("--output-dir", required=True, help="Output directory for screen artifacts")
    parser.add_argument(
        "--stage",
        choices=[
            "quick",
            "sweep",
            "final",
            "tcn_gru_focused",
            "tcn_gru_focused_final",
            "transformer_focused",
            "transformer_focused_final",
            "phase_harmonic",
            "phase_harmonic_final",
            "phase_film",
            "phase_film_final",
            "history_length",
            "temporal_order",
            "all",
        ],
        default="quick",
    )
    parser.add_argument("--recipes", nargs="*", default=None, help="Optional recipe-name filter for the chosen stage")
    parser.add_argument("--config-ids", nargs="*", default=None, help="Optional config-id filter for the chosen stage")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument(
        "--include-test-eval",
        action="store_true",
        help="Evaluate test metrics during validation-only sweep stages",
    )
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = build_screen_configs(args.stage)
    if args.recipes:
        recipe_filter = set(args.recipes)
        configs = [config for config in configs if config.recipe_name in recipe_filter]
    if args.config_ids:
        config_filter = set(args.config_ids)
        configs = [config for config in configs if config.config_id in config_filter]
    if not configs:
        raise ValueError("No screen configs selected")

    default_train, default_val, default_test = _stage_sample_defaults(args.stage)
    max_train_samples = args.max_train_samples if args.max_train_samples is not None else default_train
    max_val_samples = args.max_val_samples if args.max_val_samples is not None else default_val
    max_test_samples = args.max_test_samples if args.max_test_samples is not None else default_test
    skip_test_eval = args.stage in {
        "transformer_focused",
        "phase_harmonic",
        "phase_film",
        "history_length",
        "temporal_order",
    } and not args.include_test_eval

    rows: list[dict[str, Any]] = []
    if args.dry_run:
        rows = [_config_row(config) for config in configs]
    else:
        for config in configs:
            rows.append(
                _run_config(
                    config=config,
                    split_root=args.split_root,
                    output_dir=output_dir,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    device=args.device,
                    random_seed=args.random_seed,
                    use_amp=not args.disable_amp,
                    max_train_samples=max_train_samples,
                    max_val_samples=max_val_samples,
                    max_test_samples=max_test_samples,
                    skip_test_eval=skip_test_eval,
                )
            )

    summary = pd.DataFrame(rows)
    summary_csv_path = output_dir / "temporal_backbone_screen_summary.csv"
    summary_json_path = output_dir / "temporal_backbone_screen_summary.json"
    config_path = output_dir / "temporal_backbone_screen_config.json"
    summary.to_csv(summary_csv_path, index=False)
    summary_json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "stage": args.stage,
                "split_root": args.split_root,
                "batch_size": args.batch_size,
                "device": args.device,
                "random_seed": args.random_seed,
                "max_train_samples": max_train_samples,
                "max_val_samples": max_val_samples,
                "max_test_samples": max_test_samples,
                "skip_test_eval": skip_test_eval,
                "dry_run": bool(args.dry_run),
                "selected_config_ids": [config.config_id for config in configs],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(f"summary_csv_path: {summary_csv_path}")
    print(f"summary_json_path: {summary_json_path}")
    print(f"config_path: {config_path}")


if __name__ == "__main__":
    main()
