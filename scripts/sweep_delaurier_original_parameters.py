#!/usr/bin/env python3
"""Legacy historical sweep of the pre-3b5d4ec DeLaurier prior contract.

Do not use this pipeline for new longitudinal-force analysis. It is retained
only for reproduction of dated June-2026 artifacts; active priors are resolved
through ``configs/physics/delaurier_prior_registry.yaml``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_fx_fz_correction import TARGETS, _array, _load_split, _metrics_rows

SPLITS = ("train", "val", "test")


def _parse_csv_floats(text: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in text.split(",") if part.strip())


def _bool_text(value: bool) -> str:
    return "sep_on" if value else "sep_off"


def _candidate_name(stage: str, params: dict[str, float | bool]) -> str:
    keys = (
        "twist_eta_max_deg",
        "alpha0_deg",
        "eta_s",
        "cd_f",
        "enable_separation",
        "alpha_stall_max_deg",
        "cd_cf",
        "xi",
    )
    parts = [stage]
    for key in keys:
        value = params.get(key)
        if value is None:
            continue
        text = _bool_text(bool(value)) if isinstance(value, bool) else str(value).replace("-", "m").replace(".", "p")
        parts.append(f"{key}_{text}")
    return "__".join(parts)


def _export_prior(
    *,
    python_exe: Path,
    exporter: Path,
    split_root: Path,
    metadata: Path,
    output_root: Path,
    params: dict[str, float | bool],
    chunk_size: int,
    device: str,
    reuse_existing: bool,
) -> None:
    if reuse_existing and all((output_root / f"{split}_predictions.parquet").exists() for split in SPLITS):
        return
    cmd = [
        str(python_exe),
        str(exporter),
        "--split-root",
        str(split_root),
        "--metadata",
        str(metadata),
        "--output-root",
        str(output_root),
        "--overwrite",
        "--chunk-size",
        str(int(chunk_size)),
        "--device",
        device,
        "--theta-w-deg",
        "0.0",
        "--twist-eta-max-deg",
        str(float(params["twist_eta_max_deg"])),
        "--twist-eta-limit-deg",
        str(float(params["twist_eta_max_deg"])),
        "--alpha0-deg",
        str(float(params["alpha0_deg"])),
        "--eta-s",
        str(float(params["eta_s"])),
        "--cd-f",
        str(float(params["cd_f"])),
        "--alpha-stall-max-deg",
        str(float(params["alpha_stall_max_deg"])),
        "--cd-cf",
        str(float(params["cd_cf"])),
        "--xi",
        str(float(params["xi"])),
    ]
    if bool(params.get("enable_separation", False)):
        cmd.append("--enable-separation")
    subprocess.run(cmd, check=True, cwd=exporter.parents[2])


def _evaluate_prior(*, split_root: Path, prior_root: Path, model_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in SPLITS:
        frame = _load_split(split_root, prior_root, split)
        pred = _array(frame, tuple(f"prior_{target}" for target in TARGETS))
        rows.extend(_metrics_rows(frame, pred, split=split, model=model_name))
    return rows


def _val_mean_rmse(rows: list[dict[str, object]], model_name: str) -> float:
    for row in rows:
        if row["split"] == "val" and row["model"] == model_name and row["target"] == "fx_fz_mean":
            return float(row["rmse"])
    raise ValueError(f"missing val fx_fz_mean for {model_name}")


def _run_candidate(
    *,
    python_exe: Path,
    exporter: Path,
    split_root: Path,
    metadata: Path,
    prior_root: Path,
    params: dict[str, float | bool],
    chunk_size: int,
    device: str,
    reuse_existing: bool,
    model_name: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    _export_prior(
        python_exe=python_exe,
        exporter=exporter,
        split_root=split_root,
        metadata=metadata,
        output_root=prior_root,
        params=params,
        chunk_size=chunk_size,
        device=device,
        reuse_existing=reuse_existing,
    )
    rows = _evaluate_prior(split_root=split_root, prior_root=prior_root, model_name=model_name)
    return rows, {
        "model": model_name,
        "prior_root": str(prior_root),
        "val_rmse": _val_mean_rmse(rows, model_name),
        **params,
    }


def _append_result(
    *,
    candidate_rows: list[dict[str, object]],
    all_rows: list[dict[str, object]],
    rows: list[dict[str, object]],
    summary: dict[str, object],
    stage: str,
    output_root: Path,
) -> None:
    all_rows.extend(rows)
    candidate_rows.append({"stage": stage, **summary})
    pd.DataFrame(candidate_rows).to_csv(output_root / "original_parameter_sweep_candidates.csv", index=False)
    pd.DataFrame(all_rows).to_csv(output_root / "original_parameter_sweep_metrics.csv", index=False)


def run(
    *,
    split_root: Path,
    metadata: Path,
    output_root: Path,
    exporter: Path,
    python_exe: Path,
    twist_modes: tuple[float, ...],
    alpha0_values: tuple[float, ...],
    eta_values: tuple[float, ...],
    cd_f_values: tuple[float, ...],
    stall_values: tuple[float, ...],
    cd_cf_values: tuple[float, ...],
    xi_values: tuple[float, ...],
    chunk_size: int,
    device: str,
    reuse_existing: bool,
) -> dict[str, object]:
    split_root = split_root.resolve()
    metadata = metadata.resolve()
    output_root = output_root.resolve()
    exporter = exporter.resolve()
    python_exe = python_exe.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    prior_parent = output_root / "priors"
    prior_parent.mkdir(exist_ok=True)

    all_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    base = {
        "theta_w_deg": 0.0,
        "alpha_stall_max_deg": 12.0,
        "cd_cf": 1.95,
        "xi": 0.0,
        "enable_separation": False,
    }

    for twist in twist_modes:
        for alpha0 in alpha0_values:
            for eta_s in eta_values:
                for cd_f in cd_f_values:
                    params: dict[str, float | bool] = {
                        **base,
                        "twist_eta_max_deg": float(twist),
                        "alpha0_deg": float(alpha0),
                        "eta_s": float(eta_s),
                        "cd_f": float(cd_f),
                    }
                    name = _candidate_name("attached", params)
                    rows, summary = _run_candidate(
                        python_exe=python_exe,
                        exporter=exporter,
                        split_root=split_root,
                        metadata=metadata,
                        prior_root=prior_parent / name,
                        params=params,
                        chunk_size=chunk_size,
                        device=device,
                        reuse_existing=reuse_existing,
                        model_name=name,
                    )
                    _append_result(
                        candidate_rows=candidate_rows,
                        all_rows=all_rows,
                        rows=rows,
                        summary=summary,
                        stage="attached_alpha0_eta_cd_f",
                        output_root=output_root,
                    )

    attached = pd.DataFrame(candidate_rows).sort_values("val_rmse")
    for twist in twist_modes:
        best_for_twist = attached.loc[attached["twist_eta_max_deg"].eq(float(twist))].sort_values("val_rmse").iloc[0]
        for stall in stall_values:
            for cd_cf in cd_cf_values:
                for xi in xi_values:
                    params = {
                        **base,
                        "twist_eta_max_deg": float(twist),
                        "alpha0_deg": float(best_for_twist["alpha0_deg"]),
                        "eta_s": float(best_for_twist["eta_s"]),
                        "cd_f": float(best_for_twist["cd_f"]),
                        "enable_separation": True,
                        "alpha_stall_max_deg": float(stall),
                        "cd_cf": float(cd_cf),
                        "xi": float(xi),
                    }
                    name = _candidate_name("separation", params)
                    rows, summary = _run_candidate(
                        python_exe=python_exe,
                        exporter=exporter,
                        split_root=split_root,
                        metadata=metadata,
                        prior_root=prior_parent / name,
                        params=params,
                        chunk_size=chunk_size,
                        device=device,
                        reuse_existing=reuse_existing,
                        model_name=name,
                    )
                    _append_result(
                        candidate_rows=candidate_rows,
                        all_rows=all_rows,
                        rows=rows,
                        summary=summary,
                        stage="separation_diagnostic",
                        output_root=output_root,
                    )

    candidates = pd.DataFrame(candidate_rows).sort_values("val_rmse").reset_index(drop=True)
    metrics = pd.DataFrame(all_rows)
    selected = candidates.iloc[0].to_dict()
    candidates["is_selected"] = candidates["model"].eq(str(selected["model"]))
    metrics["is_selected"] = metrics["model"].eq(str(selected["model"]))
    candidates.to_csv(output_root / "original_parameter_sweep_candidates.csv", index=False)
    metrics.to_csv(output_root / "original_parameter_sweep_metrics.csv", index=False)
    manifest = {
        "split_root": str(split_root),
        "metadata": str(metadata),
        "output_root": str(output_root),
        "exporter": str(exporter),
        "python_exe": str(python_exe),
        "targets": list(TARGETS),
        "selected": selected,
        "scope_note": (
            "Only DeLaurier aerodynamic parameters are swept. Implementation/platform "
            "adaptation parameters are fixed: theta_w_deg=0, and twist is evaluated as a fixed mode."
        ),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--exporter", type=Path, default=Path("/home/zn/IsaacLab/scripts/flapping_px4/export_delaurier_prior_predictions.py"))
    parser.add_argument("--python-exe", type=Path, default=Path("/home/zn/anaconda3/envs/env_isaaclab/bin/python"))
    parser.add_argument("--twist-modes", default="10,0")
    parser.add_argument("--alpha0-values", default="-4,0,4")
    parser.add_argument("--eta-values", default="0.25,0.65,1.05")
    parser.add_argument("--cd-f-values", default="0,0.028,0.08")
    parser.add_argument("--stall-values", default="8,12,18")
    parser.add_argument("--cd-cf-values", default="1.2,1.95,2.7")
    parser.add_argument("--xi-values", default="0,1")
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                split_root=args.split_root,
                metadata=args.metadata,
                output_root=args.output_root,
                exporter=args.exporter,
                python_exe=args.python_exe,
                twist_modes=_parse_csv_floats(args.twist_modes),
                alpha0_values=_parse_csv_floats(args.alpha0_values),
                eta_values=_parse_csv_floats(args.eta_values),
                cd_f_values=_parse_csv_floats(args.cd_f_values),
                stall_values=_parse_csv_floats(args.stall_values),
                cd_cf_values=_parse_csv_floats(args.cd_cf_values),
                xi_values=_parse_csv_floats(args.xi_values),
                chunk_size=args.chunk_size,
                device=args.device,
                reuse_existing=args.reuse_existing,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
