# C3 — 静态纵向力修正模型正式选择

日期：2026-07-27

状态：实现完成后由 clean Commit A 正式运行并在 Commit B 冻结结果。

## 1. 执行摘要

本节在正式 Stage A/Stage B 完成后填写。

## 2. Branch 与 commits

本节记录实现 commit、结果 commit、远端状态与 clean provenance。

## 3. Authoritative dataset、prior 与 artifact

输入固定为 `longitudinal_mean_wb_ratio8_20260721T140238Z_09b4bb6`，并在运行时
通过 canonical dataset registry 与 DeLaurier prior registry fail closed 解析。

## 4. Search space

搜索空间由
`configs/correction/static_correction_model_selection_v1.yaml` 在运行前冻结。

## 5. Train grouped-CV design 与 fold composition

Stage A 只读取 train；5 folds 以 `log_id` 分组并按 flight date 与 cycle count
确定性分配。正式 fold composition 与 assignment hash 在运行后填写。

## 6. Mean branch results

正式结果待 Stage A。

## 7. Waveform branch results

正式结果待 Stage A。

## 8. Weighting refinement

只对 train-CV branch shortlist 比较 equal-cycle、equal-log 与 equal-date。

## 9. Complete candidate shortlist

正式 sealed shortlist 待 Stage A。

## 10. Validation per-log results

Stage B 只能读取 sealed finalist specs；正式结果待 Stage B。

## 11. Fx selection

正式 one-SE selection reason 待 Stage B。

## 12. Fz selection

正式 one-SE selection reason 待 Stage B。

## 13. One-standard-error rule 与 selection stability

正式 all-log 与 leave-one-validation-log-out 结果待 Stage B。

## 14. Prior incremental-value verdict

正式 matched-capacity verdict 待 Stage B。

## 15. Correction magnitude 与 numerical diagnostics

正式结果待 Stage B。

## 16. Validation envelope

正式 all-validation 与 in-envelope-only 结果待 Stage B。

## 17. C4 residual artifact

正式 validation prediction/residual 路径待 Stage B。

## 18. Limitations 与 C4 readiness

本阶段不读取 test，不训练 dynamic residual 或 TCN，也不修改 tail、moment、
controller、label、split 或 IsaacLab production physics。正式 readiness 只在
完整测试和所有 strict quality checks 通过后填写。
