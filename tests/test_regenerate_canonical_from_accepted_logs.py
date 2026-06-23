from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.regenerate_canonical_from_accepted_logs import regenerate_from_accepted_logs


def test_regenerate_from_accepted_logs_writes_manifest_and_records(tmp_path, monkeypatch):
    source_log = tmp_path / "log_0.ulg"
    source_log.write_bytes(b"fake ulg")
    metadata = tmp_path / "aircraft_metadata.yaml"
    metadata.write_text("aircraft_id: flapper_01\n", encoding="utf-8")
    accepted = tmp_path / "accepted_logs.csv"
    pd.DataFrame(
        [
            {
                "log_id": "log_0",
                "source_log_path": str(source_log),
                "sample_count": 1,
            }
        ]
    ).to_csv(accepted, index=False)

    def fake_run_ulog_to_canonical(*, ulg_path, metadata_path, output_root, rate_hz):
        output_dir = Path(output_root) / "aircraft_id=flapper_01" / "log_id=log_0"
        output_dir.mkdir(parents=True)
        samples_path = output_dir / "samples.parquet"
        segments_path = output_dir / "segments.parquet"
        manifest_path = output_dir / "source_manifest.json"
        report_path = output_dir / "preprocessing_report.json"
        pd.DataFrame(
            {
                "time_s": [0.0, 0.01],
                "fx_b": [1.0, 2.0],
                "fy_b": [0.0, 0.1],
                "fz_b": [3.0, 4.0],
                "mx_b": [0.01, 0.02],
                "my_b": [0.0, 0.01],
                "mz_b": [0.03, 0.04],
            }
        ).to_parquet(samples_path, index=False)
        pd.DataFrame({"segment_id": [0]}).to_parquet(segments_path, index=False)
        manifest_path.write_text("{}", encoding="utf-8")
        report_path.write_text("{}", encoding="utf-8")
        return {
            "samples_path": samples_path,
            "segments_path": segments_path,
            "manifest_path": manifest_path,
            "report_path": report_path,
        }

    monkeypatch.setattr(
        "scripts.regenerate_canonical_from_accepted_logs.run_ulog_to_canonical",
        fake_run_ulog_to_canonical,
    )

    output = tmp_path / "out"
    result = regenerate_from_accepted_logs(
        accepted_logs_csv=accepted,
        metadata_path=metadata,
        output_root=output,
    )

    manifest = json.loads(Path(result["dataset_manifest_path"]).read_text(encoding="utf-8"))
    records = json.loads(Path(result["accepted_logs_json_path"]).read_text(encoding="utf-8"))
    accepted_frame = pd.read_csv(result["accepted_logs_csv_path"])

    assert manifest["dataset_id"] == "out"
    assert manifest["accepted_log_count"] == 1
    assert records[0]["log_id"] == "log_0"
    assert records[0]["label_valid_ratio"] == 1.0
    assert accepted_frame.loc[0, "samples_path"].endswith("samples.parquet")
