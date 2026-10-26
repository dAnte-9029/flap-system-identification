"""Verify frozen dependencies and seal only this experiment's deliverables."""
from common import *
import platform
import sys
import subprocess
import torch, scipy, pyarrow, matplotlib, pytest


def main():
    assert OUT == HERE / 'results', 'Archive only the committed experiment directory'
    receipt = json.loads((OUT / 'input_receipt.json').read_text())
    for path, expected in receipt['prior_sha256'].items():
        assert sha(ROOT / path) == expected, path
    assert sha(HERE / 'config.json') == receipt['config_sha256']
    inference = json.loads((OUT / 'inference_checks.json').read_text())
    for record in inference['checkpoints']:
        assert sha(ROOT / record['path']) == record['sha256'], record['path']
    assert sha(OUT / 'V1_predictions.npz') == inference['variant_predictions_sha256']
    source = json.loads((OUT / 'source_receipt.json').read_text())
    for path, expected in source['python_sources_sha256'].items():
        assert sha(ROOT / path) == expected, path
    for row in pd.read_csv(OUT / 'firmware_sources.csv').itertuples():
        assert row.status == 'available_exact_commit'
        assert sha(HERE / row.snapshot) == row.sha256
    for name in ['cases', 'pairs', 'cohorts', 'windows']:
        assert sha(OUT / (name + '.csv')) == sha(SWAP / 'results' / (name + '.csv'))
    verification = json.loads((OUT / 'independent_verification.json').read_text())
    assert verification['passed'] and inference['passed']
    assert '10 passed' in (HERE / 'tests.log').read_text()
    base = 'ae8598a94555c908500856cc415cf2119c396078'
    branch = 'exp/local-control-response-time-contract-ablation-v1'
    actual_branch = subprocess.check_output(['git', '-C', str(ROOT), 'branch', '--show-current'], text=True).strip()
    assert actual_branch == branch
    subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', base, 'HEAD'], check=True)
    prefix = 'experiments/local_control_response_time_contract_ablation_v1/'
    scripts = ['audit', 'prepare', 'training_distribution', 'run', 'analyze', 'verify', 'plot', 'describe']
    execution = dict(
        base_sha=base, branch=branch, worktree=str(ROOT), environment_interpreter=sys.executable,
        python=sys.version, platform=platform.platform(), gpu=inference['gpu'],
        versions={k: v.__version__ for k, v in [('numpy', np), ('pandas', pd), ('torch', torch),
                  ('scipy', scipy), ('pyarrow', pyarrow), ('matplotlib', matplotlib), ('pytest', pytest)]},
        commands=[f'{sys.executable} {prefix}{s}.py' for s in scripts] +
                 [f'{sys.executable} -m pytest -q {prefix}test_contract.py'],
        logs={s: name for s, name in zip(scripts, ['audit.log', 'prepare.log', 'distribution.log',
              'run.log', 'analysis.log', 'verification.log', 'plot.log', 'description.log'])},
        variants_run=['V0', 'V1_source_assumption'],
        not_admitted=['V1_verified_physical_alignment', 'V2_state_and_phase'],
        frozen_prior_files_rechecked=len(receipt['prior_sha256']),
        checkpoint_files_rechecked=len(inference['checkpoints']),
        index_copies_byte_identical=True, upstream_unchanged=True,
        test_read=False, weights_updated=False, dependency_changes=False, training=False,
        parameter_search=False, validation_used_to_select_parameters=False,
        plots_visually_reviewed=['fig1_direction_amplitude_gain.png', 'case_train_336.png'],
        development_notes=[
            'Exact nearest-other-flight KD-tree query independently checked against per-flight trees.',
            'Independent graph weights and piecewise-linear integral implement separate numeric checks.',
            'Plot bounding boxes corrected after visual review; prediction arrays unchanged.',
            'Fresh-output source audit uses committed exact-commit snapshots when Git/cache is unavailable.',
            'run.py refuses to overwrite saved V1 predictions; use a fresh LOCAL_ABLATION_OUTPUT.'
        ],
        publication='Containing Git commit and verified remote ref determine publication; no assumed push.'
    )
    save(HERE / 'execution.json', execution)
    files = []
    for path in sorted(HERE.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or '.pytest_cache' in path.parts or path.name == 'archive_manifest.json':
            continue
        files.append(dict(path=str(path.relative_to(HERE)), bytes=path.stat().st_size, sha256=sha(path)))
    save(HERE / 'archive_manifest.json', dict(base_sha=base, branch=branch, file_count=len(files),
         test_read=False, excludes=['archive_manifest.json', '__pycache__', '.pytest_cache'], files=files))
    print('Sealed', len(files), 'files; upstream and six checkpoints unchanged; largest file bytes', max(r['bytes'] for r in files))


if __name__ == '__main__':
    main()
