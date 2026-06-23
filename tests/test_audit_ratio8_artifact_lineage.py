from __future__ import annotations

from pathlib import Path

from scripts.audit_ratio8_artifact_lineage import audit_ratio8_artifact_lineage


def test_audit_ratio8_artifact_lineage_flags_forbidden_strings(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "manifest.json").write_text('{"source": "ratio8_v1"}', encoding="utf-8")
    (root / "stale.md").write_text("uses artifacts/20260602_old", encoding="utf-8")

    summary = audit_ratio8_artifact_lineage(
        roots=[root],
        forbidden_substrings=["20260602", "7.5"],
        output_root=tmp_path / "audit",
    )

    assert summary["pass"] is False
    assert summary["violation_count"] == 1
    assert (tmp_path / "audit" / "violations.csv").exists()


def test_audit_ratio8_artifact_lineage_passes_clean_artifact(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "manifest.json").write_text('{"source": "ratio8_v1"}', encoding="utf-8")

    summary = audit_ratio8_artifact_lineage(
        roots=[root],
        forbidden_substrings=["20260602", "7.5"],
        output_root=tmp_path / "audit",
    )

    assert summary["pass"] is True
    assert summary["violation_count"] == 0
