"""Matched local reinitialization and free-run evaluation on September windows."""
from __future__ import annotations

import numpy as np
import pandas as pd

from system_identification.data.ulg_audit import classify_maneuvers
from system_identification.models.trajectory import attitude_error_deg

ERROR_NAMES = ("position_error_m", "velocity_error_m_s", "attitude_error_deg", "body_rate_error_rad_s")


def require_history(windows):
    return windows.loc[(windows.available_history_s >= .5) &
                       (windows.start_sample_in_segment >= 25)].copy().reset_index(drop=True)


def shifted_local_windows(samples, parents, offset_steps, length_steps=50):
    """Move origin to a true observed state; retain parent identity for pairing."""
    if offset_steps < 0 or length_steps < 1:
        raise ValueError("invalid local interval")
    rows = parents.copy()
    if (offset_steps + length_steps >= rows.state_sample_count).any():
        raise ValueError("local interval exceeds parent")
    groups = {str(k): g.set_index("sample_in_log") for k,g in samples.groupby("log_id",sort=False)}
    rows["parent_window_id"] = rows.window_id
    rows["window_id"] = rows.window_id + f":reset_{offset_steps}"
    rows["start_sample_in_log"] += offset_steps
    rows["start_sample_in_segment"] += offset_steps
    rows["end_sample_in_log"] = rows.start_sample_in_log + length_steps
    rows["end_sample_in_segment"] = rows.start_sample_in_segment + length_steps
    starts, ends = [], []
    for r in rows.itertuples():
        g = groups[r.log_id].loc[r.start_sample_in_log:r.end_sample_in_log]
        if len(g) != length_steps+1 or not g.valid_core.all() or g.segment_id.nunique()!=1:
            raise ValueError("local reset crosses source boundary")
        if int(g.segment_id.iloc[0]) != r.segment_id:
            raise ValueError("cohort/source segment mismatch")
        starts.append(int(g.timestamp_us.iloc[0])); ends.append(int(g.timestamp_us.iloc[-1]))
    rows["start_timestamp_us"] = starts
    rows["end_timestamp_us"] = ends
    rows["observed_horizon_s"] = (rows.end_timestamp_us-rows.start_timestamp_us)/1e6
    rows["horizon_s"] = length_steps/50
    rows["state_sample_count"] = length_steps+1
    rows["control_step_count"] = length_steps
    return rows


def origin_metadata(samples, windows):
    lookup = samples.set_index(["log_id","sample_in_log"])
    keys = list(zip(windows.log_id,windows.start_sample_in_log))
    s = lookup.loc[keys].reset_index()
    q = s[["attitude_q_w","attitude_q_x","attitude_q_y","attitude_q_z"]].to_numpy()
    w,x,y,z = q.T
    roll = np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y))
    speed = np.linalg.norm(s[["velocity_ned_m_s_x","velocity_ned_m_s_y","velocity_ned_m_s_z"]],axis=1)
    maneuvers = classify_maneuvers(s.velocity_ned_m_s_z,roll,s.angular_velocity_body_rad_s_z,speed)
    phase = s.logged_flap_phase_rad.to_numpy()
    valid = s.valid_logged_phase.to_numpy(dtype=bool) & np.isfinite(phase)
    bins = np.full(len(s),"missing",dtype=object)
    bins[valid] = (np.floor(phase[valid]/(2*np.pi)*8).astype(int)%8).astype(str)
    return pd.DataFrame({"window_id":windows.window_id.to_numpy(),"log_id":windows.log_id.to_numpy(),
                         "maneuver_at_origin":maneuvers,"logged_phase_bin_at_origin":bins})


def prediction_errors(prediction, truth):
    """Keep nonfinite predictions as infinite errors, never silently drop failures."""
    errors = {}
    for field,name in [("position_n",ERROR_NAMES[0]),("velocity_n",ERROR_NAMES[1]),
                       ("angular_velocity_b",ERROR_NAMES[3])]:
        with np.errstate(over="ignore",invalid="ignore"):
            value = np.linalg.norm(getattr(prediction,field)-getattr(truth,field),axis=-1)
        errors[name] = np.where(np.isfinite(value),value,np.inf)
    p = prediction.quaternion_nb.reshape(-1,4)
    t = truth.quaternion_nb.reshape(-1,4)
    finite = np.isfinite(p).all(1) & (np.linalg.norm(p,axis=1)>1e-12)
    angles = np.full(len(p),np.inf)
    angles[finite] = attitude_error_deg(p[finite],t[finite])
    errors[ERROR_NAMES[2]] = angles.reshape(truth.quaternion_nb.shape[:2])
    return errors


def error_curve(errors, log_ids, dt_s, model, mode, global_offset_s):
    """Every 50 Hz endpoint, including tail errors and declared excursion thresholds."""
    rows=[]
    observed=np.cumsum(dt_s,axis=1)
    for step in range(1,dt_s.shape[1]+1):
        excursion = (errors[ERROR_NAMES[0]][:,step]>10) | (errors[ERROR_NAMES[2]][:,step]>60)
        failed = np.zeros(len(log_ids),dtype=bool)
        for name in ERROR_NAMES:
            failed |= ~np.isfinite(errors[name][:,step])
        for log in np.unique(log_ids):
            mask=log_ids==log
            row={"model":model,"mode":mode,"log_id":log,"local_horizon_s":step/50,
                 "global_horizon_s":global_offset_s+step/50,"n_windows":int(mask.sum()),
                 "observed_local_horizon_s_mean":float(observed[mask,step-1].mean()),
                 "nonfinite_fraction":float(failed[mask].mean()),
                 "large_error_fraction":float((excursion[mask]|failed[mask]).mean())}
            for name,value in errors.items():
                v=value[mask,step]
                row[name.replace("error","rmse")]=float(np.sqrt(np.mean(v*v)))
                # Nearest-rank quantile remains well-defined when some values are inf.
                row[name+"_p95"]=float(np.sort(v)[max(0,int(np.ceil(.95*len(v)))-1)])
            rows.append(row)
    return pd.DataFrame(rows)
