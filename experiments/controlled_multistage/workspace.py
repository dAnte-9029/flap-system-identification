from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'controlled_long_horizon_stage1'))
from common import ROOT, HISTORY, TAPES, OUT as STAGE1, OLD, SUITE, conditional_data, conditional_loss, checkpoint, setup, select, query
RUN = ROOT/'outputs/controlled_multistage_20261120'
CACHE3 = RUN/'data_3s'
BASELINES = ('gru','lstm','tcn','transformer')
