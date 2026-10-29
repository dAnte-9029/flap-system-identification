"""Compare exact substep schedules on real train windows against a supplied reference tree."""
import argparse,importlib.util,sys,json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--output',required=True);a=p.parse_args()
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from system_identification.models.pahbd import PAHBD,Config
from system_identification.data.pahbd import NativeWindows,ConfiguredWindows
path=Path(a.reference)/'src/system_identification/models/pahbd'
spec=importlib.util.spec_from_file_location('pahbd_reference',path/'__init__.py',submodule_search_locations=[str(path)]);ref=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ref;spec.loader.exec_module(ref)
torch.set_num_threads(1);cfg=json.loads(Path('configs/pahbd/default.json').read_text());d=NativeWindows(**cfg['data'],split='train');rows=[]
for h in cfg['data']['horizon_us']:
    torch.manual_seed(17);old=ref.PAHBD(ref.Config(**cfg['model'])).double();new=PAHBD(Config(**cfg['model'])).double();new.load_state_dict(old.state_dict())
    i=int(d.windows[(d.windows.horizon_us==h)&d.windows.shared_periodic_eligible].index[0]);w=ConfiguredWindows(d,[i],new.config)[0]
    seq=[];pred=[]
    for m in (old,new):
        trace=[];step=m.flow.step
        def traced(s,u,dt):trace.append(dt);return step(s,u,dt)
        m.flow.step=traced
        with torch.no_grad():_,out=m(w)
        seq.append(trace);pred.append(out)
    assert seq[0]==seq[1],f'substep mismatch {h}'
    errors={}
    for k in ('x','f','phi','h','variance','phase_covariance'):
        x=torch.stack([getattr(s,k) for s in pred[0]]);y=torch.stack([getattr(s,k) for s in pred[1]])
        torch.testing.assert_close(x,y,atol=1e-9,rtol=1e-9);errors[k]=float((x-y).abs().max())
    rows.append(dict(horizon=h,steps=len(seq[0]),exact_schedule=True,max_errors=errors));print(rows[-1],flush=True)
Path(a.output).write_text(json.dumps(rows,indent=2))
