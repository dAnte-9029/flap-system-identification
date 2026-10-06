"""Causal availability and diagnostic I-update inversion; no controller replay."""
import numpy as np


def available_index(data, query):
    publication=np.asarray(data['timestamp'],np.int64)
    available=np.maximum(publication,np.asarray(data.get('timestamp_sample',publication),np.int64))
    eligible=np.flatnonzero(available<=query)
    if not len(eligible):return None
    return int(eligible[np.argmax(publication[eligible])])


def infer_cycle_dt(pre,post,error,gain,limit):
    """Infer dt assuming no hidden reset/clamp/antiwindup; never a scheduler input."""
    pre,post,error,gain,limit=[np.asarray(v,float) for v in (pre,post,error,gain,limit)]
    den=np.maximum(0,1-(error/np.deg2rad(400))**2)*gain*error
    valid=(abs(den)>.005)&(abs(pre)<limit-1e-5)&(abs(post)<limit-1e-5)
    out=np.divide(post-pre,den,out=np.full_like(error,np.nan),where=valid)
    valid &= np.isfinite(out)&(out>=.002-1e-6)&(out<=.04+1e-6)
    return np.where(valid,out,np.nan)


def highpass_next_power(hidden,previous_input,input_value,alpha):
    return (alpha*(hidden+input_value-previous_input))**2
