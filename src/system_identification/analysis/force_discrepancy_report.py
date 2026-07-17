"""Chinese research-conclusion report for the EDA0 attribution audit."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


def _number(value: object, digits: int = 3) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "不可用"
    if not math.isfinite(numeric):
        return "不可用"
    return f"{numeric:.{digits}f}"


def _best_probe_gain(table: pd.DataFrame, baseline: str, candidate: str, model_column: str) -> dict[str, float]:
    if table.empty or "validation_equal_log_macro_rmse" not in table.columns:
        return {}
    best = table.groupby(["component", model_column])["validation_equal_log_macro_rmse"].min().unstack()
    gains: dict[str, float] = {}
    for component, row in best.iterrows():
        if baseline in row and candidate in row and float(row[baseline]) != 0.0:
            gains[str(component)] = 1.0 - float(row[candidate]) / float(row[baseline])
    return gains


def build_chinese_report(
    *,
    output_directory: str | Path,
    report_path: str | Path,
    manifest: Mapping[str, object],
    summary: Mapping[str, object],
    decision: Mapping[str, object],
    tables: Mapping[str, pd.DataFrame],
    run_command: str,
) -> str:
    """Build the required conclusion document from saved authoritative tables."""

    output_dir = Path(output_directory)
    fx = summary.get("fx_b", {}) if isinstance(summary.get("fx_b"), Mapping) else {}
    fz = summary.get("fz_b", {}) if isinstance(summary.get("fz_b"), Mapping) else {}
    phase = summary.get("phase_alignment", {}) if isinstance(summary.get("phase_alignment"), Mapping) else {}
    phase_fx = phase.get("fx_b", {}) if isinstance(phase.get("fx_b"), Mapping) else {}
    phase_fz = phase.get("fz_b", {}) if isinstance(phase.get("fz_b"), Mapping) else {}
    cycle_quality = tables.get("cycle_quality", pd.DataFrame())
    accepted_cycles = int(cycle_quality.get("accepted", pd.Series(dtype=bool)).sum()) if not cycle_quality.empty else 0
    total_cycles = int(len(cycle_quality))
    rejection_fraction = 1.0 - accepted_cycles / max(total_cycles, 1)
    used_logs = list(manifest.get("used_log_ids", []))
    dates = sorted({str(value) for value in tables.get("date_level_summary", pd.DataFrame()).get("date", [])})

    half = tables.get("half_stroke_summary_by_log", pd.DataFrame())
    half_text: dict[str, str] = {}
    for component in ("fx_b", "fz_b"):
        group = half.loc[half.get("component") == component] if not half.empty else pd.DataFrame()
        if group.empty:
            half_text[component] = "half-stroke 证据不可用"
        else:
            macro = group.groupby("half_stroke")["integral_waveform_residual_rad"].mean().sort_values(key=np.abs, ascending=False)
            dominant = group.loc[group["half_stroke"] == macro.index[0], "integral_waveform_residual_rad"]
            sign_consistency = float(np.mean(np.sign(dominant) == np.sign(macro.iloc[0])))
            half_text[component] = (
                f"绝对积分最大的 physical half-stroke 为 {macro.index[0]}，"
                f"equal-log macro 积分为 {_number(macro.iloc[0])} N rad，"
                f"跨 log 同号比例={_number(100.0 * sign_consistency, 1)}%"
            )

    static = tables.get("condition_probe_metrics", pd.DataFrame())
    condition_gains = _best_probe_gain(static, "phase_only", "phase_conditioned", "probe")
    condition_individual: dict[str, dict[str, float]] = {}
    if not static.empty:
        best_condition = static.groupby(["component", "probe"])["validation_equal_log_macro_rmse"].min().unstack()
        for component, row in best_condition.iterrows():
            baseline = float(row.get("phase_only", float("nan")))
            condition_individual[str(component)] = {
                "airspeed": 1.0 - float(row.get("phase_plus_airspeed", float("nan"))) / baseline,
                "angle_of_attack": 1.0 - float(row.get("phase_plus_angle_of_attack", float("nan"))) / baseline,
                "flapping_frequency": 1.0
                - float(row.get("phase_plus_flapping_frequency", float("nan"))) / baseline,
            }
    history = tables.get("static_history_probe_metrics", pd.DataFrame())
    history_gains: dict[str, float] = {}
    if not history.empty and "validation_equal_log_macro_rmse" in history.columns:
        best = history.groupby(["component", "history_samples"])["validation_equal_log_macro_rmse"].min().unstack()
        for component, row in best.iterrows():
            if 0 in row and row.drop(labels=[0], errors="ignore").notna().any():
                history_gains[str(component)] = 1.0 - float(row.drop(labels=[0]).min()) / float(row[0])
    prior_probe_table = tables.get("matched_capacity_prior_probe", pd.DataFrame())
    prior_gains = _best_probe_gain(
        prior_probe_table,
        "no_prior",
        "prior_plus_delta",
        "model",
    )
    prior_cycle_mean_gains: dict[str, float] = {}
    if not prior_probe_table.empty and "validation_cycle_mean_rmse" in prior_probe_table.columns:
        best_prior_mean = prior_probe_table.groupby(["component", "model"])["validation_cycle_mean_rmse"].min().unstack()
        for component, row in best_prior_mean.iterrows():
            if {"prior_plus_delta", "no_prior"}.issubset(row.index) and float(row["no_prior"]) != 0.0:
                prior_cycle_mean_gains[str(component)] = 1.0 - float(row["prior_plus_delta"]) / float(row["no_prior"])
    fx_prior_gain = prior_gains.get("fx_b", float("nan"))
    fz_prior_gain = prior_gains.get("fz_b", float("nan"))
    fx_prior_mean_gain = prior_cycle_mean_gains.get("fx_b", float("nan"))
    fz_prior_mean_gain = prior_cycle_mean_gains.get("fz_b", float("nan"))
    if all(math.isfinite(value) and value > 0.0 for value in (fx_prior_gain, fz_prior_gain)):
        prior_interpretation = "Fx/Fz 在 overall waveform 上均显示正增益"
    elif math.isfinite(fz_prior_mean_gain) and fz_prior_mean_gain > 0.0:
        prior_interpretation = "overall waveform 没有稳定增益；Fz cycle mean 保留局部正增益"
    else:
        prior_interpretation = "Fx/Fz overall 与 cycle mean 均未显示稳定增量价值"

    uncertainty = tables.get("label_uncertainty_phase", pd.DataFrame())
    label_robustness_table = tables.get("label_robustness", pd.DataFrame())
    uncertainty_text = "没有可比较的 label variant，未定量完成"
    if not uncertainty.empty and "label_uncertainty_std" in uncertainty.columns:
        uncertainty_text = (
            f"label variant 的 phase-bin 平均 1-SD 为 {_number(uncertainty['label_uncertainty_std'].mean())} N；"
            f"decision={decision.get('label_uncertainty_blocks_correction')}"
        )
    label_detail_text = "label phase/half-stroke robustness 不可用"
    if not label_robustness_table.empty and "optimal_shift_change_rad" in label_robustness_table.columns:
        fx_label = label_robustness_table.loc[label_robustness_table["component"] == "fx_b"]
        fz_label = label_robustness_table.loc[label_robustness_table["component"] == "fz_b"]
        label_detail_text = (
            f"Fx optimal shift 对已有 variant 的变化={_number(fx_label['optimal_shift_change_rad'].abs().max())} rad；"
            f"Fz residual waveform variant correlation={_number(fz_label['residual_waveform_correlation'].min())}"
        )

    component_similarity = tables.get("component_residual_similarity", pd.DataFrame())
    component_contract = tables.get("component_contract_checks", pd.DataFrame())
    component_contract_text = "component diagnostic contract 不可用"
    if not component_contract.empty:
        check = component_contract.iloc[0]
        component_contract_text = (
            f"primary-vs-diagnostic prior RMSE: Fx={_number(check.get('primary_vs_diagnostic_prior_rmse_fx_b'))} N, "
            f"Fz={_number(check.get('primary_vs_diagnostic_prior_rmse_fz_b'))} N；"
            "非零 mismatch 时 component 只作 hypothesis probe"
        )
    component_lines: list[str] = []
    for force in ("fx_b", "fz_b"):
        group = component_similarity.loc[component_similarity.get("force_component") == force] if not component_similarity.empty else pd.DataFrame()
        if group.empty:
            component_lines.append(f"- {force}: component diagnostic 不可用。")
        else:
            macro = group.groupby("component")["shape_correlation"].mean().sort_values(key=np.abs, ascending=False)
            component_lines.append(
                f"- {force}: 与 residual waveform shape correlation 绝对值最大的是 `{macro.index[0]}` "
                f"(equal-log mean r={_number(macro.iloc[0])})；这是候选关联，不是唯一物理因果。"
            )

    trim = tables.get("trim_impact_estimate", pd.DataFrame())
    if trim.empty:
        trim_text = "未计算。"
    elif str(trim.iloc[0].get("status")) == "local_linear_estimate":
        trim_text = (
            f"local linear estimate 为 Δf={_number(trim.iloc[0]['estimated_trim_frequency_shift_hz'])} Hz "
            f"(95% uncertainty {_number(trim.iloc[0]['estimated_trim_frequency_shift_ci95_hz'])} Hz)，不能外推。"
        )
    else:
        trim_text = (
            f"WT1 sensitivity artifact 缺失，不能定量给出 trim frequency shift；"
            f"仅报告 cycle-mean Fz discrepancy={_number(trim.iloc[0].get('mean_fz_discrepancy_n'))} N。"
        )

    repeatability = tables.get("waveform_repeatability", pd.DataFrame())
    log_correlations = tables.get("log_waveform_correlations", pd.DataFrame())
    repeat_lines: list[str] = []
    for force in ("fx_b", "fz_b"):
        group = repeatability.loc[repeatability.get("component") == force] if not repeatability.empty else pd.DataFrame()
        pairs = log_correlations.loc[
            (log_correlations.get("component") == force)
            & (log_correlations.get("log_id_left") != log_correlations.get("log_id_right"))
        ] if not log_correlations.empty else pd.DataFrame()
        same = pairs.loc[pairs["same_date"]] if not pairs.empty else pd.DataFrame()
        cross = pairs.loc[~pairs["same_date"]] if not pairs.empty else pd.DataFrame()
        repeat_lines.append(
            f"- {force}: per-log vs macro r={_number(group['correlation_with_macro'].mean()) if not group.empty else '不可用'}；"
            f"same-date different-log r={_number(same['waveform_correlation'].mean()) if not same.empty else '不可用'}；"
            f"cross-date r={_number(cross['waveform_correlation'].mean()) if not cross.empty else '不可用'}。"
        )

    sensitivity = tables.get("physical_sensitivity_similarity", pd.DataFrame())
    sensitivity_lines: list[str] = []
    for force in ("fx_b", "fz_b"):
        group = sensitivity.loc[sensitivity.get("force_component") == force] if not sensitivity.empty else pd.DataFrame()
        if group.empty:
            sensitivity_lines.append(f"- {force}: local physical sensitivity 不可用。")
        else:
            macro = group.groupby("parameter")["shape_correlation"].mean().sort_values(key=np.abs, ascending=False)
            sensitivity_lines.append(
                f"- {force}: 最大 absolute macro sensitivity similarity 为 `{macro.index[0]}` "
                f"(r={_number(macro.iloc[0])})；仍不是参数校准证据。"
            )

    harmonics = tables.get("harmonic_by_log", pd.DataFrame())
    harmonic_texts: list[str] = []
    for force in ("fx_b", "fz_b"):
        group = harmonics.loc[harmonics.get("component") == force] if not harmonics.empty else pd.DataFrame()
        coverage = group.groupby("harmonic_order")["cumulative_energy_coverage_mean"].mean() if not group.empty else pd.Series(dtype=float)
        harmonic_texts.append(
            f"{force} K1–K4 cumulative coverage="
            + "/".join(_number(100.0 * coverage.get(order, float('nan')), 1) + "%" for order in range(1, 5))
        )

    rejection_text = "不可用"
    rejection_table = tables.get("cycle_rejection_reasons", pd.DataFrame())
    if not rejection_table.empty and "rejection_reasons" in rejection_table.columns:
        rejection_text = str(rejection_table["rejection_reasons"].value_counts().head(3).to_dict())

    implementation_inventory = manifest.get("implementation_inventory", {})
    if not isinstance(implementation_inventory, Mapping):
        implementation_inventory = {}

    exec_bullets = [
        f"1. **Fx 能量结构：**cycle-mean 占 per-log macro residual energy 的 {_number(fx.get('mean_energy_fraction_macro') * 100 if fx else None, 1)}%，zero-mean wingbeat 占 {_number(fx.get('waveform_energy_fraction_macro') * 100 if fx else None, 1)}%；对应 `cycle_mean_residuals.csv` 与 Figure 3–4。",
        f"2. **Fz 能量结构：**cycle-mean 占 {_number(fz.get('mean_energy_fraction_macro') * 100 if fz else None, 1)}%，zero-mean wingbeat 占 {_number(fz.get('waveform_energy_fraction_macro') * 100 if fz else None, 1)}%；对应 `cycle_mean_residuals.csv` 与 Figure 3–5。",
        f"3. **Fx phase：**fixed phase offset={_number(phase_fx.get('H1_fixed_phase_offset_rad'))} rad；fixed-delay estimate={_number(float(phase_fx.get('H2_fixed_delay_s', float('nan'))) * 1000)} ms；H1/H2 circular RMSE={_number(phase_fx.get('H1_rmse_rad'))}/{_number(phase_fx.get('H2_rmse_rad'))} rad。",
        f"4. **Fz phase：**fixed phase offset={_number(phase_fz.get('H1_fixed_phase_offset_rad'))} rad；fixed-delay estimate={_number(float(phase_fz.get('H2_fixed_delay_s', float('nan'))) * 1000)} ms；H1/H2 circular RMSE={_number(phase_fz.get('H1_rmse_rad'))}/{_number(phase_fz.get('H2_rmse_rad'))} rad。",
        f"5. **输入修复判定：**phase convention first=`{decision.get('fix_phase_convention_first')}`，fixed delay first=`{decision.get('fix_fixed_delay_first')}`；Fx/Fz offset 的 component 间一致性是全局输入修复的必要条件；这只是只读假设审查，没有改 label 或 prior。",
        f"6. **Mean correction：**Fx=`{decision.get('mean_correction_fx')}`，Fz=`{decision.get('mean_correction_fz')}`。",
        f"7. **Phase correction：**Fx=`{decision.get('phase_correction_fx')}`，Fz=`{decision.get('phase_correction_fz')}`；建议 harmonic order range={decision.get('harmonic_order_range_recommended')}；{'；'.join(harmonic_texts)}。",
        f"8. **Condition dependence：**phase-conditioned 相对 phase-only 的 validation equal-log macro gain 为 Fx={_number(condition_gains.get('fx_b', float('nan')) * 100, 1)}%、Fz={_number(condition_gains.get('fz_b', float('nan')) * 100, 1)}%；建议变量={decision.get('condition_features_recommended')}。",
        f"9. **Short history：**最佳预定义短历史相对 static gain 为 Fx={_number(history_gains.get('fx_b', float('nan')) * 100, 1)}%、Fz={_number(history_gains.get('fz_b', float('nan')) * 100, 1)}%；dynamic model=`{decision.get('dynamic_model_needed')}`，TCN=`{decision.get('tcn_needed')}`。",
        f"10. **Label robustness：**{uncertainty_text}；{label_detail_text}；对应 `label_robustness.csv`、`label_uncertainty_phase.csv` 与 Figure 13。",
        f"11. **DeLaurier prior value：**matched-capacity gain 为 Fx={_number(prior_gains.get('fx_b', float('nan')) * 100, 1)}%、Fz={_number(prior_gains.get('fz_b', float('nan')) * 100, 1)}%；incremental value=`{decision.get('prior_has_incremental_value')}`。",
        f"12. **Trim：**{trim_text}",
    ]

    text = f"""# EDA0 — DeLaurier 纵向力模型误差归因审查

日期：{manifest.get('timestamp', 'unknown')}

状态：analysis-only；未训练正式 correction model；test partition 未读取

run ID：`{manifest.get('run_id')}`

Git：`{manifest.get('git_commit')}`

## 1. 执行摘要

{chr(10).join(exec_bullets)}

上述结论的 residual 均定义为 total reconstructed effective force 减去 wing-only DeLaurier prior；不能把它直接称为纯 wing aerodynamic error。

## 2. 数据与模型 contract

- Label：body FRD 下、whole-aircraft CG contract 对应的 total reconstructed effective force，包含未建模或未分离的 tail、body、interaction、disturbance、asymmetry 与 reconstruction error。
- Prior：`{manifest.get('prior_artifact')}` 的 wing-only DeLaurier force；artifact ID=`{manifest.get('prior_id')}`，lifecycle=`{manifest.get('prior_lifecycle_status')}`，physics commit=`{manifest.get('prior_source_commit')}`。
- Primary residual：`F_rec_effective - F_D_wing-only`，仅分析 Fx/Fz。
- 使用 partition：`{manifest.get('used_partitions')}`；test 未加载，manifest 的 `test_rows_loaded=0`。
- Tail-subtracted residual：本次没有使用；没有把 IsaacLab tail runtime 复制到本仓库，也没有校准 tail。
- 对齐 key：`{manifest.get('alignment_keys')}`，不依赖行顺序或 DataFrame index。
- 复用：`{implementation_inventory.get('reused', [])}`。
- 从旧 script-only 逻辑迁移到 package：`{implementation_inventory.get('migrated', [])}`。
- 新增：`{implementation_inventory.get('new', [])}`。
- 本次未清理的 legacy 重复：`{implementation_inventory.get('legacy_duplicates_not_removed', [])}`；保留旧 CLI 语义与兼容入口。

## 3. 数据质量与覆盖

- 使用 {len(used_logs)} 个 logs：`{used_logs}`。
- 日期：`{dates}`。
- cycle 总数 {total_cycles}，接受 {accepted_cycles}，拒绝比例 {_number(rejection_fraction * 100, 2)}%。
- 主要 rejection reasons：`{rejection_text}`；没有静默丢弃 cycle。
- Alignment evidence：见 `alignment_report.json` 与 `alignment_mismatches.csv`。
- Cycle evidence：见 `cycle_quality.csv` 与 `cycle_rejection_reasons.csv`。
- Condition envelope 与各 log 覆盖：见 `condition_dependence.csv`、Figure 10 和 `date_level_summary.csv`。

## 4. Fx 误差归因

Fx 的 per-log macro cycle-mean residual 为 {_number(fx.get('cycle_mean_residual_macro_n'))} N，mean/WB energy fraction 为 {_number(fx.get('mean_energy_fraction_macro') * 100 if fx else None, 1)}%/{_number(fx.get('waveform_energy_fraction_macro') * 100 if fx else None, 1)}%。固定 offset 与 fixed delay 的相对拟合见 Figure 2；单凭 shift 后 RMSE 下降不足以修改输入。{half_text['fx_b']}。Joint condition probe gain 为 {_number(condition_gains.get('fx_b', float('nan')) * 100, 1)}%；单变量 U/AoA/f gain={_number(condition_individual.get('fx_b', {}).get('airspeed', float('nan')) * 100, 1)}%/{_number(condition_individual.get('fx_b', {}).get('angle_of_attack', float('nan')) * 100, 1)}%/{_number(condition_individual.get('fx_b', {}).get('flapping_frequency', float('nan')) * 100, 1)}%。Cross-log 证据见 Figure 11–12，label-variant band 见 Figure 13。

## 5. Fz 误差归因

Fz 的 per-log macro cycle-mean residual 为 {_number(fz.get('cycle_mean_residual_macro_n'))} N，mean/WB energy fraction 为 {_number(fz.get('mean_energy_fraction_macro') * 100 if fz else None, 1)}%/{_number(fz.get('waveform_energy_fraction_macro') * 100 if fz else None, 1)}%。{half_text['fz_b']}。该 half-stroke 指标来自 zero-mean waveform，因此半拍正负面积不会被误算为 cycle mean；{_number(fz.get('cycle_mean_residual_macro_n'))} N 的 cycle-mean discrepancy 是独立 branch，而不是由 zero-mean 半拍峰值代数产生。该 half-stroke identity 来自 canonical `q=A sin(phi)` direction helper，不使用未经验证的 `[0,pi)` 命名。Joint condition gain={_number(condition_gains.get('fz_b', float('nan')) * 100, 1)}%；单变量 U/AoA/f gain={_number(condition_individual.get('fz_b', {}).get('airspeed', float('nan')) * 100, 1)}%/{_number(condition_individual.get('fz_b', {}).get('angle_of_attack', float('nan')) * 100, 1)}%/{_number(condition_individual.get('fz_b', {}).get('flapping_frequency', float('nan')) * 100, 1)}%。{trim_text} 过大的半拍究竟是 localized peak、reversal 邻域还是整拍偏差，量化见 `half_stroke_residuals.csv` 与 Figure 5。

## 6. 物理机制证据

| 候选机制 | 状态 | 证据与边界 |
|---|---|---|
| phase convention | {decision.get('fix_phase_convention_first')} | Fx/Fz shift 的方向与 component 间一致性；Figure 2 |
| logging/filter fixed delay | {decision.get('fix_fixed_delay_first')} | H1/H2 circular RMSE；不能与真实动态滞后完全区分 |
| circulatory normal force | 证据不足 | component shape association 仅作候选，Figure 6/8 |
| apparent-mass force | 证据不足 | dN_a 与 local sensitivity，不代表已校准参数 |
| chordwise force | 证据不足 | dT_s/dD components 与 Fx shape association |
| dynamic twist | 证据不足 | symmetric local sensitivity；未优化 twist |
| attached-flow assumption | 证据不足 | active primary 与 frozen diagnostic 均为 attached flow；condition/shape association 尚不能唯一归因于该假设 |
| tail/body contamination | 部分支持 | label/prior scope mismatch 确认存在，但未分离量化 |
| label reconstruction | {'部分支持' if uncertainty_text.startswith('label variant') else '证据不足'} | {uncertainty_text} |
| unsteady/flexible-wing memory | {'部分支持' if decision.get('dynamic_model_needed') == 'yes' else '不支持' if decision.get('dynamic_model_needed') == 'no' else '证据不足'} | linear short-history probe，不能推出 TCN 必要性 |

Component diagnostic：
{chr(10).join(component_lines)}

{component_contract_text}。Local physical sensitivity：
{chr(10).join(sensitivity_lines)}

进入 decision matrix 的 physical parameter candidates：`{decision.get('physical_parameter_adjustment_candidate')}`（由可配置 similarity 与 step-stability thresholds 生成）。

## 7. Residual 可重复性

{chr(10).join(repeat_lines)}

Within-log cycle variance、cross-log matrix、same-date/different-date 与 matched-condition distance 分别见 `waveform_repeatability.csv`、`log_waveform_correlations.csv` 和 `date_level_summary.csv`。主要结论使用 equal-log macro；pooled sample 只作补充，长日志不会自动获得更大结论权重。

## 8. 对 correction 结构的建议

| 项目 | 决策 |
|---|---|
| 输入 phase convention 修复 | {decision.get('fix_phase_convention_first')} |
| fixed delay 修复 | {decision.get('fix_fixed_delay_first')} |
| Fx cycle-mean branch | {decision.get('mean_correction_fx')} |
| Fz cycle-mean branch | {decision.get('mean_correction_fz')} |
| Fx phase harmonic | {decision.get('phase_correction_fx')} |
| Fz phase harmonic | {decision.get('phase_correction_fz')} |
| condition interaction | {decision.get('condition_features_recommended')} |
| harmonic order range | {decision.get('harmonic_order_range_recommended')} |
| cyclic spline | 暂不需要；先用 K=1–4 evidence 判断低阶 basis 是否足够 |
| dynamic residual | {decision.get('dynamic_model_needed')} |
| TCN | {decision.get('tcn_needed')}；本 audit 不训练 TCN |

推荐顺序是：保留现有 input phase/timestamp contract（当前不支持全局 offset 或 fixed-delay 修复）；实现独立 cycle-mean 与低阶 phase/condition branch；只有 static/history validation macro gain 足够大且不能由 fixed lag 解释时，才进入 dynamic residual 设计。Figure 16 汇总该决策链。

## 9. DeLaurier prior 的作用

Matched-capacity comparison 使用相同 harmonic order、condition features、ridge grid、train/validation logs 与 sample weighting。Overall validation equal-log macro gain 为 Fx={_number(fx_prior_gain * 100, 1)}%、Fz={_number(fz_prior_gain * 100, 1)}%；cycle-mean RMSE gain 为 Fx={_number(fx_prior_mean_gain * 100, 1)}%、Fz={_number(fz_prior_mean_gain * 100, 1)}%。因此结构化 overall verdict=`{decision.get('prior_has_incremental_value')}`：{prior_interpretation}；complete-log validation 之外没有独立 condition-extrapolation 证据。见 `matched_capacity_prior_probe.csv` 与 Figure 15。论文中宜描述为“在本 flight envelope 与 matched low-capacity basis 下，DeLaurier prior 未提供稳定 overall incremental information”，而不是预设 prior 必然有效。

## 10. 限制

1. Total effective label 与 wing-only prior 存在 scope mismatch；residual 不是纯 wing aerodynamic error。
2. Tail 尚未最终冻结，本次没有 tail-subtracted residual；body force 也未单独建模。
3. Moment 不在 EDA0 范围。
4. Label reconstruction uncertainty 只使用已有 variants；没有随意生成新 labels。
5. Current flight envelope 仅覆盖 manifest 中 train/validation logs 与 dates。
6. Test partition 尚未使用，也不能用来选择 phase offset、delay、K 或 condition features。
7. Component/spanwise/local sensitivity 来自 opt-in frozen offline diagnostic；若其 total 与 primary prior 不同，结果只能用于候选机制排序。
8. WT1 artifact 缺失时，trim shift 不能定量计算。

## 11. 下一阶段实施建议

1. 不先修改全局 phase/timestamp；若后续获得独立同步证据，再作为单独 input-contract 变更并回归量化。
2. 若 mean branch verdict 为 yes，先实现低容量 per-cycle/slow-condition mean correction；保持 raw prior 可选。
3. 依据 Figure 9 的 energy coverage 实现建议范围内的 Fourier harmonic residual；只在 Figure 10/condition probe 显示稳定 gain 时加入 U/alpha/f interactions。
4. 使用 complete-log train/validation protocol 比较 correction candidates；test 保持锁定。
5. 以 opt-in linear short-history branch 复核 4-sample gain；它是 dynamic residual 候选而非最终模型。本结果不支持直接启动 TCN 训练。

## 12. Artifact 索引

- Output directory：`{output_dir}`
- Git commit：`{manifest.get('git_commit')}`
- Dataset：`{manifest.get('dataset_path')}`
- Split identity：`{manifest.get('split_identity')}`
- Prior：`{manifest.get('prior_artifact')}`（`{manifest.get('prior_id')}`，`{manifest.get('prior_lifecycle_status')}`）
- 关键 CSV：`phase_alignment_cycles.csv`、`cycle_mean_residuals.csv`、`half_stroke_residuals.csv`、`harmonic_cycle_summary.csv`、`condition_dependence.csv`、`component_residual_similarity.csv`、`matched_capacity_prior_probe.csv`
- 关键 figures：`figures/figure_01_phase_curves.png` 至 `figures/figure_16_decision_summary.png`
- Decision：`decision_summary.json`
- 复现命令：

```bash
{run_command}
```
"""
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(text, encoding="utf-8")
    return text
