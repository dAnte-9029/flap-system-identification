"""Causal rolling histories and crossed hidden/context offline diagnostics."""
import numpy as np
import torch
from system_identification.data.reference_windows import extract_window

VARIANTS=['rolling_current','legacy_current','rolling_legacy','legacy_legacy']


def rolling_windows(arrays,origin):
    inputs=[];labels=[]
    for offset in range(50):
        x,y=extract_window(arrays,origin+offset,history=26,horizon=1)
        x['history_command']=np.array(arrays['actuator_commands'][origin+offset-25:origin+offset+1],copy=True)
        x['future_command']=np.array(arrays['actuator_commands'][origin+offset:origin+offset+1],copy=True)
        inputs.append(x);labels.append(y)
    return ({k:np.stack([x[k] for x in inputs]) for k in inputs[0]},
            {k:np.stack([y[k] for y in labels]) for k in labels[0]})


@torch.no_grad()
def crossed_step(model,x,legacy_hidden,legacy_input):
    """New history encoder x old recursive-oracle hidden; current x old context.

    Only rolling_current is native rolling one-step inference. Other branches
    mix latent/context representations for offline localization, not deployment.
    """
    captured={}
    def encoder_hook(module,args,out):captured['hidden']=out[1][0]
    def cell_hook(module,args,out):captured['input']=args[0]
    def head_hook(module,args,out):captured['raw']=out
    hooks=[model.encoder.register_forward_hook(encoder_hook),model.cell.register_forward_hook(cell_hook),model.head.register_forward_hook(head_hook)]
    try: native=model(**x)
    finally:
        for h in hooks:h.remove()
    assert x['dt_s'].shape[1]==1
    result={'rolling_encoded_hidden':captured['hidden'].numpy(),'current_input':captured['input'].numpy()}
    for name,h,inp in [('rolling_current',captured['hidden'],captured['input']),
                       ('legacy_current',legacy_hidden,captured['input']),
                       ('rolling_legacy',captured['hidden'],legacy_input),
                       ('legacy_legacy',legacy_hidden,legacy_input)]:
        after=model.cell(inp,h)
        raw=model.head(torch.cat([after,inp],-1))
        derivative=raw.clamp(-6.,6.)*model.derivative_scale
        rates=x['initial_rates_b']+derivative[:,3:]*x['dt_s'][:,0,None]
        if name=='rolling_current':
            torch.testing.assert_close(raw,captured['raw'],rtol=0,atol=0)
            torch.testing.assert_close(rates,native['rates_b'][:,0],rtol=0,atol=0)
        for key,value in [('rates_b',rates),('derivative',derivative),('raw_head',raw),('hidden_after',after)]:
            result[name+'_'+key]=value.numpy()
    return result
