#!/usr/bin/env python3
"""Audit ratio-8 artifacts for stale lineage strings in lightweight text files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
}


def _iter_text_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            if root.suffix.lower() in TEXT_SUFFIXES:
                files.append(root)
            continue
        if not root.exists():
            raise FileNotFoundError(root)
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
                files.append(path)
    return sorted(files)


def audit_ratio8_artifact_lineage(
    *,
    roots: list[Path],
    forbidden_substrings: list[str],
    output_root: Path,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    files = _iter_text_files(roots)
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        for forbidden in forbidden_substrings:
            if forbidden in text:
                lines = text.splitlines()
                for index, line in enumerate(lines, start=1):
                    if forbidden in line:
                        rows.append(
                            {
                                "path": str(path),
                                "line": int(index),
                                "forbidden": forbidden,
                                "excerpt": line[:300],
                            }
                        )
    violations = pd.DataFrame(rows, columns=["path", "line", "forbidden", "excerpt"])
    violations.to_csv(output_root / "violations.csv", index=False)
    summary = {
        "roots": [str(root) for root in roots],
        "forbidden_substrings": list(forbidden_substrings),
        "files_scanned": int(len(files)),
        "violation_count": int(len(violations)),
        "pass": bool(len(violations) == 0),
        "violations_csv": str(output_root / "violations.csv"),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--forbidden-substrings", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    forbidden = [item for item in args.forbidden_substrings.split(",") if item]
    summary = audit_ratio8_artifact_lineage(
        roots=args.roots,
        forbidden_substrings=forbidden,
        output_root=args.output_root,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
