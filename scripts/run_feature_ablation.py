#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from system_identification.training import run_ablation_study


def _parse_hidden_sizes(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("hidden sizes must contain at least one integer")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run feature ablations for effective wrench regression")
    parser.add_argument("--split-root", required=True, help="Dataset split root containing train/val/test parquet files")
    parser.add_argument("--output-dir", required=True, help="Output directory for ablation artifacts")
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="Optional ablation variant names. Default runs all built-in variants.",
    )
    parser.add_argument("--hidden-sizes", type=_parse_hidden_sizes, default=(256, 256), help="Comma-separated MLP hidden sizes")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability")
    parser.add_argument("--batch-size", type=int, default=4096, help="Training batch size")
    parser.add_argument("--max-epochs", type=int, default=50, help="Maximum training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW weight decay")
    parser.add_argument("--early-stopping-patience", type=int, default=8, help="Validation patience in epochs")
    parser.add_argument("--device", default="auto", help="Training device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count")
    parser.add_argument("--disable-amp", action="store_true", help="Disable CUDA automatic mixed precision")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional train subsample size")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Optional val subsample size")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Optional test subsample size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_ablation_study(
        split_root=args.split_root,
        output_dir=args.output_dir,
        variant_names=args.variants,
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
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
