"""Read exact firmware objects locally, optionally fetch missing source snapshots.

This never changes the PX4 repository. The output is a provenance artifact,
not an assertion that missing/reconstructed firmware identities are known.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.error
import urllib.request

SOURCES = (
    "src/modules/wing_phase/WingPhase.cpp",
    "src/drivers/encoder/as5600/AS5600.cpp",
    "msg/versioned/VehicleLocalPosition.msg",
    "msg/versioned/VehicleAttitude.msg",
    "msg/versioned/VehicleAngularVelocity.msg",
    "msg/versioned/ActuatorMotors.msg",
    "msg/versioned/ActuatorServos.msg",
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repository", default="/home/zn/PX4-Autopilot")
    p.add_argument("--fetch-missing", action="store_true")
    args = p.parse_args()
    inventory = json.loads(args.inventory.read_text())
    commits = sorted({r["firmware"] for r in inventory if r["firmware"]})
    args.output.mkdir(parents=True, exist_ok=True)

    def read(item):
        commit, source = item
        r = subprocess.run(["git", "-C", args.repository, "show", f"{commit}:{source}"],
                           capture_output=True)
        data = r.stdout if r.returncode == 0 else None
        origin = "local_git_object"
        url = f"https://raw.githubusercontent.com/jiashunjiang2022/PX4-Autopilot/{commit}/{source}"
        error = None
        if data is None and args.fetch_missing:
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    data = response.read()
                origin = "exact_commit_https"
            except (urllib.error.URLError, TimeoutError) as exc:
                error = str(exc)
        if data is None:
            return dict(commit=commit, source=source, status="unavailable", error=error)
        dest = args.output / commit / source
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return dict(commit=commit, source=source, status="verified_exact_commit",
                    sha256=hashlib.sha256(data).hexdigest(), origin=origin, url=url)

    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(read, [(c, s) for c in commits for s in SOURCES]))
    (args.output / "source_manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"files": len(records), "available": sum(
        r["status"] == "verified_exact_commit" for r in records)}, indent=2))


if __name__ == "__main__":
    main()
