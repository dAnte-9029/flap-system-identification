"""Checkpoint and completion policy for the five extended flight fits."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

COMMON = 49_152
MAXIMUM = 131_072
INTERVAL = 512
WINDOW = 16
MIN_LR = 0.000125
MIN_GAIN = 0.001
RELATIVE = Path("experiments/autopilot_flight_dynamics")


def stopping_evidence(history):
    """The reported LR is the one used to train the block ending at that row."""
    recent = history[-WINDOW:]
    earlier = history[:-WINDOW]
    early_best = min((row["dev_mse"] for row in earlier), default=None)
    recent_best = min(row["dev_mse"] for row in recent)
    gain = None if early_best is None else early_best - recent_best
    floor = len(recent) == WINDOW and all(row["lr"] == MIN_LR for row in recent)
    step = history[-1]["step"]
    eligible = step >= COMMON and floor and gain is not None and gain <= MIN_GAIN
    reason = "budget_cap" if step == MAXIMUM else "validation_plateau" if eligible else None
    return dict(step=step, reason=reason, earlier_best=early_best, recent_best=recent_best,
                improvement=gain, last_16_at_min_lr=floor)


def evaluate_checkpoint(run_dir, stage):
    """Run the unchanged official evaluator in a separate process."""
    source = run_dir / "source"
    checkpoint = run_dir / stage
    env = dict(os.environ, RA_SPLIT="validation", RA_DATA_VERSION="velocity_rate_multistep_v1",
               RA_MODEL_CODE=str(source), RA_MODEL=str(checkpoint / "model.pt"))
    with tempfile.TemporaryDirectory(dir=checkpoint) as scratch:
        env.update(RA_METRICS=str(Path(scratch) / "metrics.json"), RA_OUTPUT=scratch)
        subprocess.run([sys.executable, str(source / RELATIVE / "runner.py"), "evaluate"],
                       env=env, check=True)
        for name in ("flight_mse.json", "metrics.json"):
            (Path(scratch) / name).replace(checkpoint / name)


def on_monitor(history, model):
    import torch

    run_dir = Path(os.environ["EXTENDED_RUN"])
    row = history[-1]
    decision = stopping_evidence(history)
    with (run_dir / "monitor.jsonl").open("a") as stream:
        stream.write(json.dumps({**row, **decision}) + "\n")
    if row["step"] == COMMON:
        folder = run_dir / "step49152"
        temporary = folder / "model.partial"
        torch.save(model.state_dict(), temporary)
        temporary.replace(folder / "model.pt")
        try:
            evaluate_checkpoint(run_dir, "step49152")
        except subprocess.CalledProcessError as error:
            print("Midpoint evaluation deferred to the end of training:", error, flush=True)
    if decision["reason"]:
        (run_dir / "stop.json").write_text(json.dumps(decision, indent=2) + "\n")
    return bool(decision["reason"])
