from pathlib import Path
import json
HERE=Path(__file__).resolve().parent
rows=[]
for p in sorted((HERE/'folds').glob('fold*/training/*/seed17/training.jsonl')):
 lines=p.read_text().splitlines()
 if lines:
  r=json.loads(lines[-1]);rows.append({k:r[k] for k in ['fold','model','epoch','updates','validation_flight_loss','best_epoch']})
print(json.dumps(dict(completed_runs=len(list((HERE/'folds').glob('fold*/training/*/seed17/summary.json'))),expected_runs=12,completed_epochs=sum(r['epoch'] for r in rows),completed_epoch_updates=sum(r['updates'] for r in rows),runs=rows),indent=2))
