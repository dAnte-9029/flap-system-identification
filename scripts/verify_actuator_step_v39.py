"""Freeze and execute exact persistent-step parity on original-train OOF only."""
import json
import platform
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from actuator_training_v30 import model_from
from audit_controller_support_v26 import verify_parent
from system_identification.models.actuator_command_step import ActuatorCommandStepper

OUT = base.ROOT/'docs/analysis/results/control_windows_v39'
DATA = OUT.parent/'control_windows_v29'
PARENTS = [DATA, *(OUT.parent/f'control_windows_v{n}' for n in [30, 35, 38])]


def read(path):
    return json.loads(path.read_text())


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    for parent in PARENTS:
        verify_parent(parent)
    runs = []
    for stage in [30, 35]:
        for fold in range(5):
            for seed in [17, 23, 42]:
                sub = f'fold{fold}/' + ('joint_increment/' if stage == 35 else '') + f'seed{seed}'
                root = OUT.parent/f'control_windows_v{stage}'/sub
                checkpoint = root/'final.pt'
                assert base.sha(checkpoint) == read(root/'status.json')['checkpoint_sha256']
                runs.append(dict(stage=stage, fold=fold, seed=seed,
                    checkpoint=str(checkpoint.relative_to(base.ROOT)), checkpoint_sha256=base.sha(checkpoint),
                    saved_predictions=str((root/'oof_predictions.npz').relative_to(base.ROOT)),
                    saved_predictions_sha256=base.sha(root/'oof_predictions.npz'),
                    normalization_sha256=base.sha(DATA/f'fold{fold}/normalization.npz')))
    OUT.mkdir(exist_ok=False)
    code_names = ['src/system_identification/models/actuator_command_step.py',
                  'tests/test_actuator_command_step.py', 'scripts/verify_actuator_step_v39.py']
    protocol = dict(stage='v39 persistent actuator-model step parity', status='frozen_before_execution',
        code={name:base.sha(base.ROOT/name) for name in code_names},
        parents={str(p.relative_to(base.ROOT)):base.sha(p/'delivery_manifest.json') for p in PARENTS},
        runs=runs, scope='All30 frozen v30/v35 OOF models, 407 originaltrain parents per stage/seed. No fitting, labels, developmentvalidation, sealed or reserved.',
        tolerance=dict(rtol=0, atol=0, requirement='Exact float32 equality on CPU, same <=64 batch layout as stored native predictions; hidden state also exact.'),
        sequence=dict(history=26, steps=50, dt_s=.02, command='postallocation motor0/servo012 at origin+k predicts state at origin+k+1',
            phase='native phase=anchor=0 at initialization, then phase+=2pi*fixed_frequency*dt',
            frequency='history_state[-1,11] held', airspeed='history_state[-1,12:14] held',
            previous_command='audit metadata only; native decoder has no previous-command slot'),
        checks=['native and archived output parity all50', 'native GRU hidden state all50',
                'serialized state after17 steps, fresh adapter restore and exact continuation',
                'independent interleaved sessions, all commands after step25 perturbed by0.125',
                'unchanged first25 outputs and nonzero suffix response', 'checkpoint and fit normalization hashes'],
        environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, device='cpu', threads=1),
        model_closed_loop=False, model_promoted=False, validation_accessed=False, sealed_test_accessed=False, reserved_accessed=False)
    base.write(OUT/'protocol.json', protocol)
    dataset = read(DATA/'dataset_manifest.json')
    base.write(OUT/'dataset_manifest.json', dict(dataset_id=dataset['dataset_id'],
        authority=str((DATA/'dataset_manifest.json').relative_to(base.ROOT)), authority_sha256=base.sha(DATA/'dataset_manifest.json'),
        source_manifest=dataset, partitions=['original_train_only'], labels_loaded=False,
        phase_contract=read(OUT.parent/'control_windows_v38/protocol.json')['phase_contract'],
        frequency_contract=read(OUT.parent/'control_windows_v38/protocol.json')['frequency_contract']))
    assert base.sha(DATA/'windows.npz') == dataset['arrays_sha256']
    with np.load(DATA/'windows.npz') as z:
        x = {k[6:]:torch.tensor(z[k]) for k in z.files if k.startswith('input_')}
    assert len(x['history_state']) == 407
    rows = []; seen = {}; state_count = 0
    for run in runs:
        stage, fold, seed = (run[key] for key in ['stage', 'fold', 'seed'])
        dest = OUT/f'v{stage}_fold{fold}_seed{seed}'
        dest.mkdir()
        ids = read(DATA/f'fold{fold}/manifest.json')['query_indices']
        seen.setdefault((stage, seed), []).extend(ids)
        model = model_from(base.ROOT/run['checkpoint'])
        with np.load(DATA/f'fold{fold}/normalization.npz') as z:
            for key in z.files:
                np.testing.assert_array_equal(getattr(model, key).numpy(), z[key].astype(np.float32))
        adapter = ActuatorCommandStepper(model)
        restorer = ActuatorCommandStepper(model)
        assert adapter.model_id == restorer.model_id
        results = {key:[] for key in ['velocity_n', 'quaternion', 'rates_b']}
        saved = np.load(base.ROOT/run['saved_predictions'])
        np.testing.assert_array_equal(saved['array_index'], ids)
        snapshots = {}
        for start in range(0, len(ids), 64):
            batch_ids = ids[start:start+64]
            batch = {key:value[batch_ids] for key,value in x.items()}
            history = {key:value for key,value in batch.items() if key not in ['future_command', 'dt_s']}
            hidden = []
            hook = model.cell.register_forward_hook(lambda module, args, output: hidden.append(output.clone()))
            with torch.no_grad():
                native = model(**batch)
            hook.remove()
            altered = dict(batch)
            altered['future_command'] = batch['future_command'].clone()
            altered['future_command'][:,25:] += .125
            with torch.no_grad():
                native_altered = model(**altered)
            for key in native:
                np.testing.assert_array_equal(native[key].numpy(), saved[key][start:start+len(batch_ids)])
            a = adapter.initialize(**history)
            b = adapter.initialize(**history)
            output = {key:[] for key in native}; hidden_max = 0.
            for step in range(50):
                # Independent branch first on odd steps, second on even steps.
                def advance_a():
                    return (adapter if step < 17 else restorer).step(a, batch['future_command'][:,step], batch['dt_s'][:,step])
                def advance_b():
                    return adapter.step(b, altered['future_command'][:,step], batch['dt_s'][:,step])
                if step % 2:
                    b, ob = advance_b(); a, oa = advance_a()
                else:
                    a, oa = advance_a(); b, ob = advance_b()
                torch.testing.assert_close(a.hidden, hidden[step], rtol=0, atol=0)
                hidden_max = max(hidden_max, float((a.hidden-hidden[step]).abs().max()))
                for key in native:
                    torch.testing.assert_close(oa[key], native[key][:,step], rtol=0, atol=0)
                    torch.testing.assert_close(ob[key], native_altered[key][:,step], rtol=0, atol=0)
                    if step < 25:
                        torch.testing.assert_close(oa[key], ob[key], rtol=0, atol=0)
                    output[key].append(oa[key])
                if step == 16:
                    snapfile = dest/f'state_step17_batch{start}.pt'
                    torch.save(adapter.snapshot(a), snapfile)
                    a = restorer.restore(torch.load(snapfile, map_location='cpu', weights_only=True))
                    snapshots[str(start)] = dict(array_index=batch_ids, path=snapfile.name, sha256=base.sha(snapfile))
                    state_count += len(batch_ids)
            assert not torch.equal(native['rates_b'][:,25:], native_altered['rates_b'][:,25:])
            for key in native:
                value = torch.stack(output[key], 1).numpy()
                results[key].append(value)
                rows.append(dict(stage=stage, fold=fold, seed=seed, batch_start=start, windows=len(batch_ids),
                    field=key, max_abs_error=float(np.abs(value-native[key].numpy()).max()),
                    hidden_max_abs_error=hidden_max, restore_exact=True, interleaved_exact=True, prefix_exact=True))
        saved.close()
        np.savez_compressed(dest/'step_predictions.npz', array_index=ids, **{key:np.concatenate(value) for key,value in results.items()})
        base.write(dest/'snapshots.json', snapshots)
        base.write(dest/'verification.json', dict(**run, status='passed', model_id=adapter.model_id, windows=len(ids),
            restored_windows=len(ids), every50_outputs_and_hidden_exact=True, independent_sessions_exact=True,
            prefix_causality_exact=True, changed_suffix_response=True, normalization_exact=True))
        assert base.sha(base.ROOT/run['checkpoint']) == run['checkpoint_sha256']
        pd.DataFrame(rows).to_csv(OUT/'parity.csv', index=False)
        base.write(OUT/'status.json', dict(status='running', completed_models=len(list(OUT.glob('v*/verification.json'))), total_models=30))
        print(f'v{stage} fold{fold} seed{seed}: {len(ids)} OOF windows, 50 steps exact', flush=True)
    for ids in seen.values():
        np.testing.assert_array_equal(sorted(ids), np.arange(407))
    assert state_count == 407*6
    for path,digest in protocol['code'].items():
        assert base.sha(base.ROOT/path) == digest
    for parent in PARENTS:
        verify_parent(parent)
    base.write(OUT/'verification.json', dict(status='passed', checkpoints=30, unique_parents=407,
        model_window_runs=state_count, model_window_steps=state_count*50,
        perturbed_session_steps=state_count*50, restored_window_states=state_count,
        all_outputs_hidden_and_archives_exact=True, checkpoint_normalization_hashes_unchanged=True,
        no_future_measured_state_or_hidden_injection=True, model_closed_loop=False, model_promoted=False,
        validation_accessed=False, sealed_test_accessed=False, reserved_accessed=False))
    base.write(OUT/'status.json', dict(status='complete_verification_pending_report', completed_models=30))


if __name__ == '__main__':
    main()
