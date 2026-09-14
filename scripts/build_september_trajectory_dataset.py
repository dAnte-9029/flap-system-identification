#!/usr/bin/env python3
"""Build the explicit September Step 1 train/validation dataset."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from system_identification.data.september_trajectory import build_september_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/data/trajectory_september_v2.yaml")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    manifest = build_september_dataset(args.config, ROOT, args.output_root)
    print(json.dumps(manifest["partitions"], indent=2))


if __name__ == "__main__":
    main()
