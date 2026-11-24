"""Finalize once the comparison artifacts are atomically usable, including its trained pipeline checks."""
import json
import subprocess
import time
from workspace import RUN
from report import write_report
from verify_final import main as verify


if __name__=='__main__':
    # The training queue owns figure and score output. Finish only after it has exited.
    while subprocess.run(['tmux','-L','controlled-multi','has-session','-t','train3'],capture_output=True).returncode==0:
        time.sleep(20)
    if not (RUN/'rmse_horizons.png').exists() or not (RUN/'trained3_pipeline.json').exists():
        raise RuntimeError('training queue exited before final scoring; see member and queue logs')
    write_report()
    verify()
    (RUN/'completed.json').write_text(json.dumps(dict(training_members=11,updates_per_member=2976,
                                                       representative=json.loads((RUN/'freeze.json').read_text())['representative'])))
