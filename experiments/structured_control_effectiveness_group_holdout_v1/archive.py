"""Hash the completed experiment without modifying any upstream artifact."""
from common import *
import datetime

def main():
    assert json.loads((OUT/'training_completion.json').read_text())['completed']==27
    assert json.loads((OUT/'independent_verification.json').read_text())['passed']
    assert (HERE/'report_zh.md').is_file()
    assert '14 passed' in (HERE/'tests_final.log').read_text()
    save(OUT/'completion_receipt.json',dict(passed=True,training_runs=27,epochs_each=24,final_tests_passed=14,independent_verification_sha256=sha(OUT/'independent_verification.json'),final_tests_sha256=sha(HERE/'tests_final.log'),initial_pipeline_failure_preserved='pipeline_execution.json and tests_final_initial_failure.log' if (HERE/'tests_final_initial_failure.log').exists() else None,resolution='Corrected an erroneous expected excluded-training-background count from 7 to 5; the other 2 are non-primary validation background pairs. Membership was checked against immutable fold/pair indices. No training, predictions, folds or primary statistics changed.' if (HERE/'tests_final_initial_failure.log').exists() else 'No historical count-assertion failure in this reproduction; final membership assertions derive from frozen indices.',excluded_train_background_pairs=5,excluded_nonprimary_validation_pairs=2,additional_training_after_fixed_budget=0,final_result_sha256={name:sha(OUT/name) for name in ['factual_points.csv.gz','factual_summary.csv','internal_points.csv.gz','internal_summary.csv','paired_summary.csv','heldout_trajectory.csv.gz']},test_artifact_read=False))
    files=[]
    for p in sorted(HERE.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.name=='archive_manifest.json':continue
        assert p.suffix!='.tmp',p
        files.append(dict(path=str(p.relative_to(HERE)),bytes=p.stat().st_size,sha256=sha(p)))
    save(HERE/'archive_manifest.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),base_sha='f1324dfaaa807297d83b1d246cf21e362d5140bf',branch='exp/structured-control-effectiveness-group-holdout-v1',files=files,test_artifact_read=False,checkpoint_files=sum(x['path'].endswith('.pt') for x in files)))
    assert sum(x['path'].endswith('.pt') for x in files)==54
    print(len(files),'files; bytes',sum(x['bytes'] for x in files),'54 best/last checkpoints')

if __name__=='__main__':main()
