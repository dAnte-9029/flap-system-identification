import numpy as np
import pytest
from system_identification.evaluation.trajectory_consistency import integration_residual, metric_tables, shifted_velocity_residual


def test_trapezoid_consistency_with_nonuniform_time_and_position_correction():
    t=np.array([0.,.01,.04,.06]);v=np.zeros((1,4,3));v[0,:,0]=2*t+3
    p=np.zeros_like(v);p[0,:,0]=t*t+3*t
    np.testing.assert_allclose(integration_residual(p,v,np.diff(t)[None]),0,atol=1e-15)
    p[0,2:,1]+=.2
    r=integration_residual(p,v,np.diff(t)[None])
    np.testing.assert_allclose(r.sum(axis=1),[[0,.2,0]],atol=1e-15)


def test_equal_log_and_vector_metric_are_not_component_or_pooled_rms():
    x=np.array([[[3.,4.,0.]],[[0.,0.,0.]],[[0.,0.,0.]]])
    per,macro=metric_tables({'p':x},['a','b','b'],[1])
    assert per.loc[per.log_id=='a','vector_rmse'].item()==5
    assert macro.loc[macro.aggregation=='equal_log','vector_rmse'].item()==2.5
    assert macro.loc[macro.aggregation=='pooled_windows','vector_rmse'].item()==pytest.approx(5/np.sqrt(3))
    x[0,0,0]=np.nan
    _,macro=metric_tables({'p':x},['a','b','b'],[1])
    assert np.isinf(macro.vector_rmse).all()
    assert (macro.nonfinite_windows==1).all()


def test_lag_uses_identical_interior_and_recovers_known_velocity_offset():
    t=np.linspace(0,2,101)[None];p=np.zeros((1,101,3));v=p.copy()
    p[0,:,0]=t[0]**2;v[0,:,0]=2*(t[0]-.1)
    zero,d0=shifted_velocity_residual(p,v,t,0)
    aligned,d1=shifted_velocity_residual(p,v,t,.1)
    np.testing.assert_allclose(d0,d1)
    assert abs(zero[0,0])>.3
    np.testing.assert_allclose(aligned,0,atol=1e-12)
    with pytest.raises(ValueError): integration_residual(p,v,-np.diff(t))
