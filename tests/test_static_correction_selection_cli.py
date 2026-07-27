from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


PYTHON = "/home/zn/anaconda3/envs/flap-train-gpu/bin/python"
STAGE_A = "scripts/build_static_correction_train_cv_shortlist.py"
STAGE_B = "scripts/evaluate_static_correction_validation_finalists.py"


@pytest.mark.parametrize("script", [STAGE_A, STAGE_B])
def test_cli_help_smoke(script: str) -> None:
    result = subprocess.run([PYTHON, script, "--help"], text=True, capture_output=True)
    assert result.returncode == 0
    assert "--correction-ready-root" in result.stdout


@pytest.mark.parametrize(
    ("script", "partition", "message"),
    [
        (STAGE_A, "validation", "Stage A only accepts"),
        (STAGE_A, "test", "Stage A only accepts"),
        (STAGE_B, "train", "Stage B only accepts"),
        (STAGE_B, "test", "Stage B only accepts"),
    ],
)
def test_wrong_stage_partition_fails_before_file_reads(
    script: str, partition: str, message: str, tmp_path: Path
) -> None:
    command = [
        PYTHON,
        script,
        "--correction-ready-root",
        str(tmp_path / "missing-artifact"),
        "--config",
        str(tmp_path / "missing-config"),
        "--partition",
        partition,
        "--output-root",
        str(tmp_path / "output"),
    ]
    if script == STAGE_B:
        command.extend(
            [
                "--shortlist",
                str(tmp_path / "missing-shortlist"),
                "--selected-bundle-root",
                str(tmp_path / "bundles"),
            ]
        )
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode != 0
    assert message in result.stderr
    assert not (tmp_path / "output").exists()


def test_stage_b_requires_selected_bundle_root() -> None:
    result = subprocess.run([PYTHON, STAGE_B, "--help"], text=True, capture_output=True)
    assert "--selected-bundle-root" in result.stdout
