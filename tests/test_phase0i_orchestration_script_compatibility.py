from __future__ import annotations

import sys

import pandas as pd

import scripts.run_baseline_comparison as baseline_cli
import scripts.run_feature_ablation as ablation_cli
import scripts.run_training_diagnostics as diagnostics_cli
import scripts.train_baseline_torch as training_cli
import system_identification.artifacts.io as artifact_io
import system_identification.evaluation.diagnostics as diagnostics
import system_identification.training as training
import system_identification.training.data_preparation as data_preparation
import system_identification.training.orchestration as orchestration
import system_identification.training.recipes as recipes


MODULE_SYMBOLS = {
    data_preparation: ("DEFAULT_TARGET_COLUMNS", "_set_random_seed", "_load_split_frame"),
    recipes: ("fit_torch_sequence_regressor", "fit_torch_rollout_regressor", "fit_torch_regressor"),
    diagnostics: (
        "DEFAULT_REGIME_BIN_SPECS",
        "ACCELERATION_INPUT_COLUMNS",
        "VELOCITY_HISTORY_COLUMNS",
        "ANGULAR_VELOCITY_HISTORY_COLUMNS",
        "ALPHA_BETA_HISTORY_COLUMNS",
        "_training_audit_flags",
        "_sequence_arrays_for_bundle",
        "_sequence_arrays_with_metadata_for_bundle",
        "_rollout_arrays_for_bundle",
        "_targets_for_bundle",
        "prediction_metadata_frame_for_bundle",
        "predict_model_bundle",
        "evaluate_model_bundle",
        "evaluate_model_bundle_by_log",
        "evaluate_model_bundle_by_regime_bins",
        "run_diagnostic_evaluation",
    ),
    artifact_io: ("_save_training_curves", "_save_pred_vs_true_plot", "_save_residual_hist_plot"),
    orchestration: (
        "LEAKAGE_RESISTANT_BASELINE_PROTOCOL",
        "BASELINE_COMPARISON_RECIPES",
        "DEFAULT_FEATURE_GROUPS",
        "DEFAULT_ABLATION_VARIANTS",
        "_ordered_unique_columns",
        "resolve_ablation_variants",
        "run_training_job",
        "run_ablation_study",
        "_resolve_baseline_recipe_names",
        "_run_single_baseline_recipe",
        "_run_split_axis_baseline_recipe",
        "run_baseline_comparison",
    ),
}


def test_training_reexports_all_phase0i_canonical_objects():
    assert sum(len(symbols) for symbols in MODULE_SYMBOLS.values()) == 37
    for module, symbols in MODULE_SYMBOLS.items():
        for symbol in symbols:
            assert getattr(training, symbol) is getattr(module, symbol)


def test_recipe_name_dispatch_preserves_order_defaults_and_errors():
    expected_default = list(orchestration.BASELINE_COMPARISON_RECIPES)
    assert training._resolve_baseline_recipe_names(None) == expected_default
    requested = [expected_default[2], expected_default[0]]
    assert orchestration._resolve_baseline_recipe_names(requested) == requested

    try:
        orchestration._resolve_baseline_recipe_names(["not_a_recipe"])
    except ValueError as exc:
        assert "Unknown baseline comparison recipes" in str(exc)
    else:
        raise AssertionError("unknown recipes must retain the existing ValueError")


def test_diagnostics_per_log_output_schema_is_unchanged(monkeypatch):
    def fake_evaluate(bundle, frame, *, split_name, batch_size, device):
        return {
            "split": split_name,
            "sample_count": len(frame),
            "overall_mae": 1.0,
            "overall_rmse": 2.0,
            "overall_r2": 0.5,
            "per_target": {"fx_b": {"mae": 1.0, "rmse": 2.0, "r2": 0.5}},
        }

    monkeypatch.setattr(diagnostics, "evaluate_model_bundle", fake_evaluate)
    frame = pd.DataFrame({"log_id": ["b", "a", "b", "a"], "value": [1, 2, 3, 4]})
    result = diagnostics.evaluate_model_bundle_by_log(
        {"target_columns": ["fx_b"]},
        frame,
        split_name="val",
        min_samples=1,
        batch_size=2,
        device="cpu",
    )

    assert list(result["log_id"]) == ["a", "b"]
    assert list(result.columns) == [
        "split",
        "diagnostic_type",
        "group_column",
        "group_value",
        "val_sample_count",
        "val_overall_mae",
        "val_overall_rmse",
        "val_overall_r2",
        "val_fx_b_mae",
        "val_fx_b_rmse",
        "val_fx_b_r2",
        "log_id",
    ]


def test_artifact_writer_keeps_caller_path_and_history_schema(tmp_path):
    history = pd.DataFrame(
        {
            "epoch": [1.0, 2.0],
            "train_loss": [1.0, 0.5],
            "val_loss": [1.2, 0.6],
            "val_overall_rmse": [2.0, 1.0],
            "val_overall_mae": [1.5, 0.8],
            "val_fx_b_r2": [0.1, 0.5],
        }
    )
    output_path = tmp_path / "training_curves.png"
    training._save_training_curves(history, output_path)

    assert training._save_training_curves is artifact_io._save_training_curves
    assert output_path.exists()
    assert output_path.name == "training_curves.png"
    assert output_path.stat().st_size > 0


def test_selected_core_scripts_are_thin_canonical_callers(monkeypatch):
    assert training_cli.run_training_job is orchestration.run_training_job
    assert baseline_cli.run_baseline_comparison is orchestration.run_baseline_comparison
    assert baseline_cli.BASELINE_COMPARISON_RECIPES is orchestration.BASELINE_COMPARISON_RECIPES
    assert ablation_cli.run_ablation_study is orchestration.run_ablation_study
    assert diagnostics_cli.run_diagnostic_evaluation is diagnostics.run_diagnostic_evaluation
    assert diagnostics_cli.DEFAULT_REGIME_BIN_SPECS is diagnostics.DEFAULT_REGIME_BIN_SPECS

    monkeypatch.setattr(
        sys,
        "argv",
        ["train_baseline_torch.py", "--split-root", "split", "--output-dir", "run"],
    )
    args = training_cli.parse_args()
    assert args.split_root == "split"
    assert args.output_dir == "run"
    assert args.model_type == "mlp"
    assert args.batch_size == 4096
    assert args.max_epochs == 50
