"""Train-only exact-gradient preflight and one fixed CRM weight calibration."""
import argparse
import copy
import json
import time
import numpy as np
import pandas as pd
import torch
from pacld_crm_v2 import *

def compare(a,b):
    return dict(max_abs=float(np.max(abs(a-b))),relative_l2=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-20)))

def equal_state(a,b):
    if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal_state(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal_state(x,y) for x,y in zip(a,b))
    return a==b

def preflight(bundle,store,ids):
    seed();a,oa=model_and_optimizer(bundle);initial=copy.deepcopy(oa.state_dict())
    for _ in range(2):
        b,ob=model_and_optimizer(bundle);assert equal_state(a.state_dict(),b.state_dict());assert equal_state(initial,ob.state_dict());del b,ob
    del a,oa
    # FP64 checks isolate algorithmic equivalence from production mixed-precision rounding.
    model,_=model_and_optimizer(bundle,precision='fp64');model.eval();small=ids[:4]
    ix,batch,plan=next(store.chunks(small,model,4))
    out=model(batch,1.);m,p=store.moments(model,out['belief'],plan,ix)
    four_macro_valid=int(out['macro_valid'].sum());four_pair_counts=torch.bincount(batch.pairs['batch'],minlength=4).cpu().tolist()
    values=dict(full=float(out['full'].mean().detach()),macro=float(out['macro'].mean().detach()),M=m.detach().cpu().numpy())
    gs=[vector(model,out['total'],True),vector(model,m[0].square(),True),vector(model,m[1].square())]
    detached=[model.zero_grad(set_to_none=True)]
    calcs=[np.zeros_like(gs[0]) for _ in range(3)];losses=np.zeros(2)
    for ii,ba,pl in store.chunks(small,model,2):
        r=model(ba,1.);mm,_=store.moments(model,r['belief'],pl,ii);f=len(ii)/len(small)
        losses+=f*np.array([float(r['full'].mean().detach()),float(r['macro'].mean().detach())])
        calcs[0]+=f*vector(model,r['total'],True)
        calcs[1]+=f*2*values['M'][0]*vector(model,mm[0],True)
        calcs[2]+=f*2*values['M'][1]*vector(model,mm[1])
    errors={name:compare(g,c) for name,g,c in zip(['FULL_Macro','normal_M2','future_M2'],gs,calcs)}
    assert all(v['relative_l2']<1e-8 for v in errors.values()),errors
    np.testing.assert_allclose(losses,[values['full'],values['macro']],rtol=1e-10,atol=1e-10)
    masks=module_masks(model)
    assert np.linalg.norm(gs[0][masks['belief']])>0
    for v in gs[1:]:
        assert np.max(abs(v[masks['belief']]))==0
        for name in ['coupling','decoder','slow','equilibrium','relaxation']:assert np.linalg.norm(v[masks[name]])>0
    # Future residual values are consumed only after future.x exists. Permuting them cannot change prediction.
    with torch.no_grad():
        mm,p1=store.moments(model,out['belief'],plan,small)
        old=store.frame.loc[small,'R_future100'].copy();store.frame.loc[small,'R_future100']=old.to_numpy()[::-1]
        mm2,p2=store.moments(model,out['belief'],plan,small)
        store.frame.loc[small,'R_future100']=old.to_numpy()
        assert torch.equal(p1,p2)
    del model,out,batch,gs,calcs;torch.cuda.empty_cache()
    # One disposable 32-window optimizer smoke, not part of any formal arm.
    seed();model,opt=model_and_optimizer(bundle);snapshot={n:p.detach().clone() for n,p in model.named_parameters()}
    torch.cuda.reset_peak_memory_stats();start=time.monotonic();smoke=ids[:32]
    M=first_pass(model,store,smoke,32)
    val,valid,_=accumulated_backward(model,store,smoke,32,'B1',1.,M)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);opt.step();torch.cuda.synchronize()
    assert any(not torch.equal(snapshot[n],p) for n,p in model.named_parameters())
    assert all(torch.isfinite(p).all() for p in model.parameters())
    elapsed=time.monotonic()-start
    result=dict(status='PASS',equal_three_model_states=True,equal_three_optimizer_states=True,optimizer_initial_state_entries=len(initial['state']),shared_sampling_sha256=sha(OUT/'shared_sampling_indices.csv'),effective_batch_contract=512,gradient_equivalence_fp64=errors,full_macro_loss_equivalence=losses.tolist(),four_window_macro_valid_windows=four_macro_valid,four_window_macro_pair_counts=four_pair_counts,smoke_macro_valid_windows=int(valid),moment_belief_gradient_zero=True,dynamics_decoder_reachable=True,future_residual_permutation_prediction_exact=True,smoke=dict(windows=32,disposable_optimizer_steps=1,seconds=elapsed,windows_per_second=32/elapsed,peak_cuda_bytes=torch.cuda.max_memory_allocated(),grad_norm=float(norm),clip_triggered=bool(norm>10),production_microbatch=32),validation_test_opened=False,original_model_sources_changed=False,checkpoint_unchanged=sha(CHECKPOINT)==SHA)
    from system_identification.data.pacld import open_windows
    frozen=json.loads(Path('configs/pacld/default.json').read_text())
    assert bundle['config']['model']==frozen['model'] and bundle['config']['data']==frozen['data']
    identity=open_windows(bundle['config'],'train').identity()
    for key in ['manifest_sha256','index_sha256','windows','split']:assert identity[key]==bundle['data_identity']['train'][key]
    result.update(data_identity_train=identity,original_config_matches=True)
    write_json(OUT/'preflight.json',result);del model,opt;torch.cuda.empty_cache();return result

def calibration(bundle,store,cal,micro):
    model,_=model_and_optimizer(bundle);model.eval();masks=module_masks(model);initial={k:v.clone() for k,v in model.state_dict().items()};vectors=[];rows=[];start=time.monotonic()
    for bid,g in cal.groupby('batch'):
        ids=g.cohort_index.to_list();dim=sum(p.numel() for p in model.parameters());grad=[np.zeros(dim) for _ in range(3)];M=np.zeros(2);total=0.
        torch.cuda.reset_peak_memory_stats();t=time.monotonic()
        for j,(ix,batch,plan) in enumerate(store.chunks(ids,model,micro)):
            out=model(batch,1.);mm,_=store.moments(model,out['belief'],plan,ix);f=len(ix)/512
            M+=f*mm.detach().cpu().numpy();total+=f*float(out['total'].detach())
            grad[0]+=f*vector(model,out['total'],True);grad[1]+=f*vector(model,mm[0],True);grad[2]+=f*vector(model,mm[1])
            print('calibration',bid,'windows',min((j+1)*micro,512),'/512','elapsed_s',round(time.monotonic()-t,1),flush=True)
        grad[1]*=2*M[0];grad[2]*=2*M[1]
        assert all(np.isfinite(x).all() for x in grad)
        np.savez_compressed(LOCAL/f'calibration_gradients_{bid}.npz',base=grad[0],normal=grad[1],future=grad[2],M=M)
        vectors.append(grad)
        for arm,j in [('B1',1),('B2',2)]:
            for module,mask in masks.items():
                base=grad[0][mask];crm=grad[j][mask]
                rows.append(dict(batch=int(bid),group=arm,module=module,base_norm=float(np.linalg.norm(base)),moment_norm=float(np.linalg.norm(crm)),cosine=cosine(base,crm),M=float(M[j-1]),base_loss=total,seconds=time.monotonic()-t,peak_cuda_bytes=torch.cuda.max_memory_allocated()))
    assert equal_state(initial,model.state_dict())
    stats=pd.DataFrame(rows);dy=stats[stats.module=='dynamics'];lambdas={}
    reasons=[];stability=[];cfg=json.loads((OUT/'training_contract.json').read_text())['calibration'];gates=cfg['stop_rules']
    for arm,j in [('B1',1),('B2',2)]:
        a=dy[(dy.group==arm)&(dy.batch==0)].iloc[0]
        lam=.25*a.base_norm/a.moment_norm if a.moment_norm>0 else None;lambdas[arm]=lam
        if lam is None or not np.isfinite(lam) or lam>gates['max_lambda']:reasons.append(arm+': zero/nonfinite/excessive lambda')
        for row in dy[dy.group==arm].itertuples():
            ratio=row.moment_norm/row.base_norm
            if ratio<gates['min_moment_to_base_norm']:reasons.append(f'{arm}: batch{row.batch} near-zero moment gradient')
            weighted=None if lam is None else lam*ratio
            if weighted is None or not gates['weighted_ratio_range'][0]<=weighted<=gates['weighted_ratio_range'][1]:reasons.append(f'{arm}: batch{row.batch} weighted gradient ratio outside preregistered range')
        for aa in range(3):
            for bb in range(aa+1,3):
                c=cosine(vectors[aa][j][masks['dynamics']],vectors[bb][j][masks['dynamics']]);stability.append(dict(group=arm,batch_a=aa,batch_b=bb,cosine=c))
                if c is None or c<gates['minimum_pairwise_cosine']:reasons.append(f'{arm}: opposite Dynamics M2 gradients in batches {aa}/{bb}')
    stats['lambda']=stats.group.map(lambdas);stats['weighted_ratio']=stats['lambda']*stats.moment_norm/stats.base_norm
    clipping=[]
    for bid,gg in enumerate(vectors):
        for arm,j in [('B1',1),('B2',2)]:
            norm=float(np.linalg.norm(gg[0]+lambdas[arm]*gg[j]));clipping.append(dict(batch=bid,group=arm,preclip_norm=norm,clip_threshold=10.,would_clip=norm>10))
    result=dict(status='BLOCKED' if reasons else 'PASS',lambdas=lambdas,ratio_target=.25,fit_batch=0,diagnostics=stats.replace({np.nan:None}).to_dict('records'),stability=stability,clipping=clipping,reasons=reasons,elapsed_s=time.monotonic()-start,parameters_unchanged=True,validation_test_opened=False,formal_optimizer_updates=0)
    write_json(OUT/'loss_calibration.json',result)
    write_json(OUT/'result_summary.json',dict(status='blocked_before_formal_training' if reasons else 'calibrated',reasons=reasons,formal_updates={'B0':0,'B1':0,'B2':0},H1='UNTESTED',H2='UNTESTED',H3='UNTESTED'))
    print(json.dumps(dict(status=result['status'],lambdas=lambdas,reasons=reasons,stability=stability),indent=2),flush=True)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    contract=initialize();seed();bundle=torch.load(CHECKPOINT,map_location='cpu',weights_only=False);store=Store(bundle)
    cal=pd.read_csv(LOCAL/'calibration_indices.csv');store.prepare(cal.cohort_index.to_list())
    if not (OUT/'preflight.json').exists():preflight(bundle,store,cal.cohort_index.to_list())
    if a.preflight_only:return
    if (OUT/'loss_calibration.json').exists():raise FileExistsError('preserve the one registered calibration; no rerun')
    result=calibration(bundle,store,cal,contract['microbatch'])
    if result['status']!='PASS':raise SystemExit(2)

if __name__=='__main__':main()
