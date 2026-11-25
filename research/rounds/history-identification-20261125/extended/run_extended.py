"""Launch or score one preregistered flight fit from its pinned Git sources."""
import argparse
import difflib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from policy import COMMON, INTERVAL, MAXIMUM, MIN_GAIN, MIN_LR, RELATIVE, WINDOW, evaluate_checkpoint, stopping_evidence

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
MODELS = {
    "baseline": "8c3a3e585f96e31e893a7287a0006b73daf999f3",
    "history": "50e3ae0bedf34b0c85bc07b9ccd18d090c13ee79",
    "current": "149a384c01e2f33b787c87f21f94b0ca51fd7fc6",
    "high": "363dfc9ebb99ef2133f2dde65d9cdd797ebd7c00",
    "low": "505a14493a150fec7b61938a9871a80f65ba5452",
}
SEED = 17


def git_file(commit, name):
    return subprocess.check_output(["git", "-C", str(PROJECT), "show", f"{commit}:{RELATIVE / name}"])


def resolve_model(value):
    if value in MODELS:
        return value
    matches = [name for name, commit in MODELS.items() if len(value) >= 7 and commit.startswith(value)]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Expected a registered model name or commit, got {value!r}")


def prepare_sources(run_dir, model):
    folder = run_dir / "source" / RELATIVE
    folder.mkdir(parents=True)
    for name in ("runner.py", "intervals.py", "data.json"):
        data = git_file(MODELS["baseline"], name)
        if git_file(MODELS[model], name) != data:
            raise ValueError(f"{model} changed the shared protocol file {name}")
        (folder / name).write_bytes(data)
    (folder / "model.py").write_bytes(git_file(MODELS[model], "model.py"))
    shutil.copyfile(HERE / "policy.py", folder / "policy.py")
    original = (folder / "runner.py").read_text()
    old = '''            tail = [v["dev_mse"] for v in history[-rule["window"]:]]
            if step >= rule["min_steps"] and len(tail) == rule["window"] and max(tail) - min(tail) <= rule["max_range"]:
                break'''
    new = '''            from policy import on_monitor
            if on_monitor(history, model):
                break'''
    if original.count(old) != 1:
        raise ValueError("The frozen runner stopping block does not match")
    extended = original.replace(old, new)
    (folder / "extended_runner.py").write_text(extended)
    patch = difflib.unified_diff(original.splitlines(True), extended.splitlines(True),
                                 fromfile="runner.py", tofile="extended_runner.py")
    (run_dir / "runner.diff").write_text("".join(patch))
    for path in folder.glob("*.py"):
        compile(path.read_bytes(), str(path), "exec")
    return folder


def check():
    with tempfile.TemporaryDirectory() as tmp:
        for model in MODELS:
            folder = prepare_sources(Path(tmp) / model, model)
            data = json.loads((folder / "data.json").read_text())
            assert data["dataset"] == "velocity_rate_multistep_v1"
            assert set(data["split"]) == {"train", "validation"}
            for part, counts in (("train", (121, 126237)), ("validation", (39, 44524))):
                rec = data["split"][part]
                assert (rec["flights"], rec["windows"]) == counts and Path(rec["file"]).is_file()
            print(model, MODELS[model], "sources and prepared cache recipe OK")
    def rows(step):
        return [dict(step=i, dev_mse=.1, lr=MIN_LR) for i in range(INTERVAL, step + 1, INTERVAL)]
    assert stopping_evidence(rows(COMMON - INTERVAL))["reason"] is None
    sufficient = rows(COMMON)
    assert stopping_evidence(sufficient)["reason"] == "validation_plateau"
    sufficient[-WINDOW]["lr"] *= 2
    assert stopping_evidence(sufficient)["reason"] is None
    sufficient[-WINDOW]["lr"] = MIN_LR
    sufficient[-1]["dev_mse"] -= 2 * MIN_GAIN
    assert stopping_evidence(sufficient)["reason"] is None
    assert stopping_evidence(rows(MAXIMUM))["reason"] == "budget_cap"
    print("Floor-LR, gain, common-budget and hard-cap cases OK")


def run(run_dir, model, gpu):
    run_dir.mkdir(parents=True, exist_ok=False)
    for stage in ("step49152", "final"):
        (run_dir / stage).mkdir()
    folder = prepare_sources(run_dir, model)
    recipe = dict(model=model, model_commit=MODELS[model], runner_commit=MODELS["baseline"],
                  seed=SEED, cuda_visible=gpu, common_updates=COMMON, cap_updates=MAXIMUM,
                  monitor_interval=INTERVAL, tail_length=WINDOW, lr_floor=MIN_LR, min_gain=MIN_GAIN)
    (run_dir / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    subprocess.run([sys.executable, str(folder / "runner.py"), "preflight"], check=True)
    subprocess.run([sys.executable, str(folder / "extended_runner.py"), "train"], check=True)
    if not (run_dir / "step49152" / "metrics.json").is_file():
        evaluate_checkpoint(run_dir, "step49152")
    evaluate_checkpoint(run_dir, "final")
    report = json.loads((run_dir / "train_info.json").read_text())
    stop = json.loads((run_dir / "stop.json").read_text())
    assert stop["reason"] in ("budget_cap", "validation_plateau") and stop["step"] == report["steps"]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="action", required=True)
    commands.add_parser("check", help="CPU-only source, cache metadata and stop-policy checks")
    for action in ("run", "evaluate"):
        sub = commands.add_parser(action)
        sub.add_argument("model", help="Registered model name or Git commit")
        sub.add_argument("--gpu", required=True, choices=("0", "1"))
        sub.add_argument("--output", type=Path, help="Alternative per-model directory")
        if action == "evaluate":
            sub.add_argument("stage", choices=("step49152", "final"))
    args = cli.parse_args()
    if args.action == "check":
        return check()
    model = resolve_model(args.model)
    run_dir = (args.output or HERE / "runs" / model).resolve()
    os.environ.pop("PYTHONPATH", None)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, HIP_VISIBLE_DEVICES=args.gpu,
                      ROCR_VISIBLE_DEVICES=args.gpu, PYTHONDONTWRITEBYTECODE="1",
                      RA_SPLIT="train", RA_DATA_VERSION="velocity_rate_multistep_v1",
                      RA_SEED=str(SEED), RA_STEPS=str(MAXIMUM), EXTENDED_RUN=str(run_dir),
                      RA_MODEL=str(run_dir / "final" / "model.pt"),
                      RA_TRAIN_INFO=str(run_dir / "train_info.json"),
                      RA_TRAINING_COMPLETION=json.dumps(dict(mode="validation_plateau", metric="dev_mse",
                                                             eval_every=INTERVAL)))
    if args.action == "run":
        return run(run_dir, model, args.gpu)
    recipe = json.loads((run_dir / "recipe.json").read_text())
    if recipe["model_commit"] != MODELS[model]:
        raise ValueError("Existing output belongs to a different model")
    evaluate_checkpoint(run_dir, args.stage)


if __name__ == "__main__":
    main()
