#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from system_identification.training import run_training_job


def _parse_hidden_sizes(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("hidden sizes must contain at least one integer")
    return values


def _parse_optional_int(raw: str) -> int | None:
    if raw.lower() in {"none", "null"}:
        return None
    return int(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline PyTorch MLP for effective wrench regression")
    parser.add_argument("--split-root", required=True, help="Dataset split root containing train/val/test parquet files")
    parser.add_argument("--output-dir", required=True, help="Output directory for model artifacts")
    parser.add_argument(
        "--feature-set",
        default=None,
        help="Named feature set: full, no_accel_no_alpha, paper_no_accel_v2, or paper_pfnn_10",
    )
    parser.add_argument("--model-type", default="mlp", choices=["mlp", "pfnn", "causal_gru", "causal_gru_asl"], help="Regressor architecture")
    parser.add_argument("--hidden-sizes", type=_parse_hidden_sizes, default=(256, 256), help="Comma-separated MLP hidden sizes")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability")
    parser.add_argument("--pfnn-expanded-input-dim", type=int, default=45, help="PFNN input expansion size")
    parser.add_argument("--pfnn-phase-node-count", type=int, default=5, help="PFNN phase-generated node count")
    parser.add_argument("--pfnn-control-points", type=int, default=6, help="PFNN cyclic Catmull-Rom control point count")
    parser.add_argument("--batch-size", type=int, default=4096, help="Training batch size")
    parser.add_argument("--max-epochs", type=int, default=50, help="Maximum training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW weight decay")
    parser.add_argument("--early-stopping-patience", type=int, default=8, help="Validation patience in epochs")
    parser.add_argument("--loss-type", default="mse", choices=["mse", "huber"], help="Training loss in scaled target space")
    parser.add_argument("--huber-delta", type=float, default=1.0, help="Huber delta in scaled target units")
    parser.add_argument("--window-mode", default="single", choices=["single", "causal", "centered"], help="Temporal feature window mode")
    parser.add_argument("--window-radius", type=int, default=0, help="Number of neighboring samples to include")
    parser.add_argument(
        "--window-feature-mode",
        default="all",
        choices=["all", "none", "phase_actuator", "phase_actuator_airdata", "no_kinematics"],
        help="Which feature groups receive temporal windows; non-windowed features keep t+0 only",
    )
    parser.add_argument("--device", default="auto", help="Training device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count")
    parser.add_argument("--disable-amp", action="store_true", help="Disable CUDA automatic mixed precision")
    parser.add_argument(
        "--target-loss-weights",
        default=None,
        help="Comma-separated target weights, e.g. fx_b=1,fy_b=0.5,fz_b=1,mx_b=0.5,my_b=1,mz_b=0.5",
    )
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional train subsample size")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Optional val subsample size")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Optional test subsample size")
    parser.add_argument("--sequence-history-size", type=int, default=64, help="Causal sequence history length")
    parser.add_argument(
        "--sequence-feature-mode",
        default="phase_actuator_airdata",
        choices=["all", "none", "phase_actuator", "phase_actuator_airdata"],
        help="Feature groups used as sequence history",
    )
    parser.add_argument(
        "--current-feature-mode",
        default="remaining_current",
        choices=["remaining_current", "all", "none"],
        help="Point features concatenated to sequence representation",
    )
    parser.add_argument("--gru-num-layers", type=int, default=1, help="Number of GRU layers for sequence models")
    parser.add_argument("--asl-hidden-size", type=int, default=128, help="ASL frequency gate hidden size")
    parser.add_argument("--asl-dropout", type=float, default=0.1, help="ASL dropout probability")
    parser.add_argument("--asl-max-frequency-bins", type=_parse_optional_int, default=None, help="ASL retained RFFT bins, or none")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_training_job(
        split_root=args.split_root,
        output_dir=args.output_dir,
        feature_set_name=args.feature_set,
        model_type=args.model_type,
        hidden_sizes=args.hidden_sizes,
        dropout=args.dropout,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        device=args.device,
        random_seed=args.random_seed,
        num_workers=args.num_workers,
        use_amp=not args.disable_amp,
        target_loss_weights=args.target_loss_weights,
        loss_type=args.loss_type,
        huber_delta=args.huber_delta,
        window_mode=args.window_mode,
        window_radius=args.window_radius,
        window_feature_mode=args.window_feature_mode,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        pfnn_expanded_input_dim=args.pfnn_expanded_input_dim,
        pfnn_phase_node_count=args.pfnn_phase_node_count,
        pfnn_control_points=args.pfnn_control_points,
        sequence_history_size=args.sequence_history_size,
        sequence_feature_mode=args.sequence_feature_mode,
        current_feature_mode=args.current_feature_mode,
        gru_num_layers=args.gru_num_layers,
        asl_hidden_size=args.asl_hidden_size,
        asl_dropout=args.asl_dropout,
        asl_max_frequency_bins=args.asl_max_frequency_bins,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
