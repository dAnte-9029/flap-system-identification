"""Reproducible independent dataset build; all raw inputs are read-only."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from system_identification.data.velocity_rate_build import build


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "configs/data/velocity_rate_multistep_v1.yaml")
    p.add_argument("--output", type=Path)
    p.add_argument("--firmware-sources", type=Path, required=True)
    args = p.parse_args()
    build(ROOT, args.config, args.output, args.firmware_sources)


if __name__ == "__main__":
    main()
