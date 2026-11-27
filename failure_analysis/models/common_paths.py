from pathlib import Path
import os
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
MAIN = Path(os.getenv("FLAP_DATA_ROOT", "/home/zn/flap-system-identification"))
OUT = MAIN / "outputs"
ROUND = PROJECT / "research/rounds/phase-module-evidence-20261125"
DIAG = MAIN / "research/rounds/periodic-motion-20261125/diagnostics"
TMP = Path(os.getenv("FLAP_POINTS", "/tmp/slow-fast-v1-points"))
TMP.mkdir(exist_ok=True, parents=True)
