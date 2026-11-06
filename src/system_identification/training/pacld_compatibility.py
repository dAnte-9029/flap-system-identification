"""Reviewed source compatibility, separate from Git and operational configuration.

Unknown source digests fail closed. A documented equivalence review may add a
new digest under the SAME semantic version; changed mathematics needs a new
version and cannot resume the old checkpoint. The registry is not auto-updated.
"""
import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ROOT/'configs/pacld/source_compatibility.json'


def _ast(source):
    tree = ast.parse(textwrap.dedent(source))
    # Comments, docstrings and source line offsets are nonsemantic.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body = node.body[1:]
    return ast.dump(tree, include_attributes=False)


def source_digests():
    from . import pacld
    from .pahbd import FlightBalancedSampler
    from ..models.pahbd.dynamics import PhasePropagator
    files = lambda names: [_ast((ROOT/name).read_text()) for name in names]
    model = files(['src/system_identification/models/pacld/'+name+'.py'
                   for name in ('__init__','types','dynamics','belief','flow')]
                  +['src/system_identification/models/pahbd/geometry.py','src/system_identification/models/pahbd/types.py'])
    model.append(_ast(inspect.getsource(PhasePropagator)))
    loss = [_ast(inspect.getsource(getattr(pacld, name))) for name in
            ('valid_pairs','state_loss','transition_residual','macro_loss','window_objective')]
    groups = dict(model=model, loss=loss,
        data=files(['src/system_identification/data/pahbd.py','src/system_identification/data/pahbd_phase.py',
                    'src/system_identification/data/pacld.py']),
        sampler=[_ast(inspect.getsource(FlightBalancedSampler))],
        optimization=files(['src/system_identification/training/pacld.py',
                            'src/system_identification/training/pacld_compatibility.py','scripts/run_pacld.py']),
        evaluation=files(['src/system_identification/evaluation/pacld.py','src/system_identification/evaluation/pahbd.py']))
    return {name:hashlib.sha256('\n'.join(parts).encode()).hexdigest() for name,parts in groups.items()}


def current_contract():
    registry = json.loads(REGISTRY.read_text())
    actual = source_digests()
    for name, digest in actual.items():
        if digest not in registry['sources'][name]['approved_sha256']:
            raise ValueError(f'unreviewed PACLD {name} source; explicit semantic/equivalence review required')
    return dict(versions={k:v['version'] for k,v in registry['sources'].items()}, sources=actual)


def check_source_contract(saved):
    current = current_contract()
    registry = json.loads(REGISTRY.read_text())
    if saved.get('versions') != current['versions'] or set(saved.get('sources',{})) != set(current['sources']):
        raise ValueError('checkpoint semantic/optimization version mismatch')
    for name,digest in saved['sources'].items():
        if digest not in registry['sources'][name]['approved_sha256']:
            raise ValueError(f'checkpoint {name} source not confirmed compatible')
    return current


def scientific_config(config):
    result = copy.deepcopy(config)
    result.pop('data_cache',None)
    for key in ('gradient_diagnostics_interval','terminal_log_interval'):
        result['training'].pop(key,None)
    return result


def optimization_rule(config):
    return dict(version='pacld_adam_global_l2_clip_v1', optimizer=config['optimizer'],
                max_grad_norm=config['training']['max_grad_norm'], norm_type=2.,
                ordering='finite loss/state -> diagnostics -> backward -> finite gradients -> clip -> Adam')
