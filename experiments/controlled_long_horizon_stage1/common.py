from pathlib import Path
import sys
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = ROOT/'outputs/controlled_long_horizon_stage1_20261119'
HISTORY = ROOT/'outputs/history_slow_fast_20261119_cache'
TAPES = ROOT/'outputs/controlled_slow_fast_20261119_tapes'
OLD = ROOT/'outputs/controlled_slow_fast_20261119_run1'
SUITE = ROOT/'outputs/controlled_baselines_20261119_run1'
sys.path.insert(0,str(HERE.parent/'controlled_slow_fast_v1'))
from fit import conditional_data, conditional_loss as original_loss, make_model
from model import trajectory_losses
from stage_query import query
from dynamics import inverse, qmul, qexp, unit, prefix_rotations, geodesic
from train import select, validation


def frozen(name):
    root = OLD if name in ('slowfast','gru') else SUITE
    ck = torch.load(root/name/'best.pt',map_location='cpu',weights_only=False)
    sd = ck['model']
    m = make_model(name,sd['mean'],sd['scale']).cuda().eval()
    m.load_state_dict(sd)
    return m,ck


def setup():
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')


def checkpoint(name):
    if name in ('c1', 'c2'):
        from harmonic import candidate_model
        ck = torch.load(OUT/name/'best.pt', map_location='cpu', weights_only=False)
        sd = ck['model']
        model = candidate_model(name, sd['mean'], sd['scale']).cuda().eval()
        model.load_state_dict(sd)
        return model, ck
    return frozen(name)


def conditional_loss(model, batch, units):
    if not hasattr(model, 'partial_acceleration'):
        return original_loss(model, batch, units)
    history, initial, y, t, valid, beat, endpoint, tape = batch
    roll = model(history, initial, tape)
    pv = query(roll, t[:, :, 0], 'pv')
    q = query(roll, t[:, :, 1], 'q')
    w = query(roll, t[:, :, 2], 'w')
    slow_v = query(roll, t[:, :, 0], 'slow_v')
    slow_w = query(roll, t[:, :, 2], 'slow_w')
    return trajectory_losses(pv,q,w,y,valid,beat,endpoint,units,slow_v,slow_w)
