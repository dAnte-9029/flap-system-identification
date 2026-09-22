"""Presentation checks; do not read held-out logs or change statistical rules."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pandas as pd
from report_paper_independent_test import table,markdown
from paper_independent_test_core import METRICS


def test_table_keeps_dates_deterministic_reference_and_missing_group():
    rows=[]
    for date in ['Sep8','Sep19']:
        for model in ['B0','H26']:
            for metric in METRICS:
                rows.append(dict(date=date,model=model,condition='Actual',group='ALL',horizon_s=.5,metric=metric,mean=1.,std=np.nan if model=='B0' else .1))
    for metric in METRICS:rows.append(dict(date='Sep8',model='H26',condition='Hold',group='high',horizon_s=.5,metric=metric,mean=np.nan,std=np.nan))
    t=table(pd.DataFrame(rows));assert len(t)==5
    assert '±' not in markdown(t[t.model=='B0 kinematic'])
    assert '1.0000 ± 0.1000' in markdown(t[t.model=='Standard GRU/H26'])
    assert 'unavailable' in markdown(t[t.group=='high'])
    assert set(t.date)=={'Sep8','Sep19'}
