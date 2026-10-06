"""Independent data-membership, kinematic summaries and fit-only gradient checks."""
import numpy as np
import pandas as pd
import torch
import train_control_windows_v2 as base
from prepare_attitude_v42 import OUT,DATA,PARENTS,read
from train_joint_increment_v34 import load_inputs,ActuatorCommandGRU
from joint_increment_v34 import objective as legacy_objective,select_local
from audit_controller_support_v26 import verify_parent


def direct_gravity(q):
    q=q/q.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    w,x,y,z=q.unbind(-1)
    return torch.stack([2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],-1)


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    assert read(OUT/'verification.json')['status']=='passed'
    p=read(OUT/'protocol.json');plan=read(OUT/'training_plan.json');batches=read(OUT/'calibration_batches.json')
    audit=pd.read_csv(OUT/'gradient_audit.csv');x,y,lx,ly=load_inputs();checks=0
    assert base.sha(OUT/'weights.json')==plan['weights_sha256']
    for run in plan['runs']:
        fold,seed=run['fold'],run['seed'];info=read(DATA/f'fold{fold}/manifest.json');fit=info['fit_indices']
        assert base.sha(base.ROOT/run['sample_plan'])==run['sample_plan_sha256']
        with np.load(base.ROOT/run['sample_plan']) as z:draws=z['array_index'];offsets=z['offset']
        with np.load(DATA/f'fold{fold}/normalization.npz') as z:stats={k:z[k] for k in z.files}
        torch.manual_seed(seed);model=ActuatorCommandGRU(stats);params=tuple(model.parameters())
        for b in [v for v in batches if (v['fold'],v['seed'])==(fold,seed)]:
            k=b['batch'];ids=np.asarray(b['parents']);np.testing.assert_array_equal(ids,draws[k]);np.testing.assert_array_equal(b['offsets'],offsets[k])
            assert k in range(4) and set(ids)<=set(fit) and not set(ids)&set(info['query_indices'])
            bx={n:v[ids] for n,v in x.items()};by={n:v[ids] for n,v in y.items()};local_x,local_y=select_local(lx,ly,ids,offsets[k],fit)
            old=legacy_objective(model,bx,by,local_x,local_y,weight=1.)[0]
            ga=torch.cat([v.reshape(-1) for v in torch.autograd.grad(old,params)])
            pred=model(**bx)['quaternion'];delta=direct_gravity(pred)-direct_gravity(by['quaternion'])
            tilt=delta.square().sum(-1).mean()/.01
            gb=torch.cat([v.reshape(-1) for v in torch.autograd.grad(tilt,params)])
            row=audit[(audit.fold==fold)&(audit.seed==seed)&(audit.batch==k)&(audit.stage=='fresh')].iloc[0]
            # Separate expression graph may round its gradients slightly differently.
            np.testing.assert_allclose([float(old.detach()),float(tilt.detach()),float(ga.norm()),float(gb.norm())],
                row[['v35_loss','tilt_loss','v35_grad_norm','tilt_grad_norm']].to_numpy(float),rtol=2e-6,atol=2e-6)
            checks+=1
    assert checks==60
    for weight in read(OUT/'weights.json'):
        a=audit[(audit.fold==weight['fold'])&(audit.stage=='fresh')];assert len(a)==12
        raw=.5*np.median(a.v35_grad_norm/a.tilt_grad_norm)
        np.testing.assert_allclose([raw,np.clip(raw,.01,10)],[weight['unclipped_weight'],weight['weight']],rtol=1e-12,atol=1e-12)
        for r in plan['runs']:
            if r['fold']==weight['fold']:assert r['tilt_weight']==weight['weight']
    table=pd.read_csv(OUT/'kinematics_per_flight.csv').set_index(['fold','variant','stage','seed','oracle','horizon','kind','metric']).sort_index()
    names=['native','true_p','true_q','true_r','true_pq','true_pr','true_qr','true_pqr','remove_window_rate_bias'];values=0
    for path in sorted((OUT/'kinematics').glob('fold*/*.npz')):
        fold=int(path.parent.name[4:]);stem=path.stem;variant,rest=stem.rsplit('_v',1);stage,s=rest.split('_seed');stage=int(stage);seed=int(s)
        ids=read(DATA/f'fold{fold}/manifest.json')['query_indices']
        with np.load(path) as z:
            np.testing.assert_array_equal(z['array_index'],ids);sq=z['angle_squared']
        for j,name in enumerate(names):
            for h in [25,50]:
                for kind,m in [('wave',sq[:,:h,j].mean((0,1))),('endpoint',sq[:,h-1,j].mean(0))]:
                    for axis,metric in enumerate(['roll','pitch','tilt']):
                        np.testing.assert_allclose(np.sqrt(m[axis]),table.loc[(fold,variant,stage,seed,name,h,kind,metric),'rmse'],rtol=1e-12,atol=1e-12);values+=1
    zero=read(OUT/'zero_weight_reproduction.json');a=torch.load(OUT/'zero_control/final.pt',map_location='cpu',weights_only=False)['state_dict']
    b=torch.load(base.ROOT/zero['baseline_checkpoint'],map_location='cpu',weights_only=False)['state_dict']
    for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    assert base.sha(OUT/'zero_control/final.pt')==zero['reproduction_sha256']
    for parent in PARENTS:verify_parent(parent)
    for name,digest in p['code'].items():assert base.sha(base.ROOT/name)==digest
    base.write(OUT/'independent_verification.json',dict(status='passed',fresh_fit_gradient_pairs_recomputed=checks,
        coefficients_recomputed=5,kinematic_metric_cells_recomputed=values,full1600_zero_state_dict_exact=True,
        query_parents_absent_from_calibration=True,candidate_training_started=False))
    print('Independent kinematic summaries, fit-only coefficients and full zero control passed',flush=True)


if __name__=='__main__':main()
