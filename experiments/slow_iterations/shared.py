from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'coupled_slow'
sys.path.insert(0,str(PRIOR))
from network import ROOT, Dynamics, TAU, wave, load_tuned
from dataset import PREV, CACHE
from fit_coupled import read, select
EXTRA=ROOT/'.worktrees/slow-motion-three-rounds/outputs/slow_iterations'
