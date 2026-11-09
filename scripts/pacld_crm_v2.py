"""Fixed-budget CRM-v2 orchestration primitives. No changes to PACLD sources."""
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import time
from functools import lru_cache
import numpy as np
import pandas as pd
import torch
from pacld_crm_v1 import CHECKPOINT, SHA, SOURCE
from system_identification.models.pacld import Config
from system_identification.models.pacld.batch import BatchedPACLD, FIELDS
from system_identification.data.pacld_batch import compile_windows, steps, pad_steps
from evaluate_pacld_a2_v2_gpu import prepare_batch

BASE=Path('experiments/phase_anchored_coupled_latent_v1')
OUT=BASE/'crm_training_comparison_v2'
V1=BASE/'control_residual_moment_v1'
LOCAL=OUT/'local'
BASE_SHA='9c15fffd24bfcacc7c57cd980cc149098cb83f82'

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));tmp.replace(path)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def seed():
    random.seed(17);np.random.seed(17);torch.manual_seed(17);torch.cuda.manual_seed_all(17);torch.set_num_threads(1)

def model_and_optimizer(bundle,device='cuda:0',precision=None):
    model=BatchedPACLD(Config(**bundle['config']['model']),precision or bundle['precision'],checkpoint_steps=True).to(device)
    model.load_state_dict(bundle['model'])
    opt=torch.optim.Adam(model.parameters(),lr=bundle['config']['optimizer']['lr'])
    if bundle.get('optimizer',{}).get('state'):opt.load_state_dict(copy.deepcopy(bundle['optimizer']))
    return model,opt

def vector(model,loss,retain=False):
    params=list(model.parameters());gs=torch.autograd.grad(loss,params,allow_unused=True,retain_graph=retain)
    return np.concatenate([(torch.zeros_like(p) if g is None else g).detach().cpu().double().numpy().ravel() for p,g in zip(params,gs)])

def module_masks(model):
    labels=[]
    for n,p in model.named_parameters():
        label=n.split('.')[2] if n.startswith('flow.dynamics.') else ('belief' if n.startswith('belief.') else 'other')
        labels.extend([label]*p.numel())
    labels=np.array(labels)
    return {k:labels==k for k in sorted(set(labels))}|{'dynamics':~np.isin(labels,['belief','other'])}

def cosine(a,b):
    norm=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/norm) if norm else None

def hierarchy_draw(frame,rng,n):
    # IID hierarchical slots: each marginal exactly date/flight/window equal.
    ds=[[(g.index.to_numpy()) for _,g in day.groupby('flight')] for _,day in frame.groupby('date')]
    result=[]
    for _ in range(n):
        fs=ds[rng.integers(len(ds))];ws=fs[rng.integers(len(fs))];result.append(int(rng.choice(ws)))
    return result

def initialize():
    if (OUT/'training_contract.json').exists():
        c=json.loads((OUT/'training_contract.json').read_text())
        # Reconstruct excluded identity tables deterministically on a fresh server checkout.
        if not (OUT/'shared_sampling_indices.csv').exists() or not (LOCAL/'calibration_indices.csv').exists():
            frame=pd.read_csv(V1/'windows.csv');LOCAL.mkdir(parents=True,exist_ok=True)
            ids=hierarchy_draw(frame,np.random.default_rng(17),64*512)
            sample=frame.loc[ids,['flight','t','date']].reset_index().rename(columns={'index':'cohort_index','flight':'flight_id','t':'decision_us'})
            sample.insert(0,'update',np.repeat(np.arange(1,65),512));sample.insert(1,'slot',np.tile(np.arange(512),64))
            if not (OUT/'shared_sampling_indices.csv').exists():sample.to_csv(OUT/'shared_sampling_indices.csv',index=False)
            ids=hierarchy_draw(frame,np.random.default_rng(np.random.SeedSequence(17,spawn_key=(1,))),3*512)
            cal=pd.DataFrame({'batch':np.repeat(np.arange(3),512),'slot':np.tile(np.arange(512),3),'cohort_index':ids})
            if not (LOCAL/'calibration_indices.csv').exists():cal.to_csv(LOCAL/'calibration_indices.csv',index=False)
        assert sha(OUT/'shared_sampling_indices.csv')==c['sampling_sha256']
        assert sha(LOCAL/'calibration_indices.csv')==c['calibration_sha256']
        return c
    assert sha(CHECKPOINT)==SHA
    bundle=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    frame=pd.read_csv(V1/'windows.csv');assert (len(frame),frame.flight.nunique(),frame.date.nunique())==(11221,114,17)
    assert not frame.duplicated(['flight','t']).any()
    LOCAL.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(17);ids=hierarchy_draw(frame,rng,64*512)
    sample=frame.loc[ids,['flight','t','date']].reset_index().rename(columns={'index':'cohort_index','flight':'flight_id','t':'decision_us'})
    sample.insert(0,'update',np.repeat(np.arange(1,65),512));sample.insert(1,'slot',np.tile(np.arange(512),64));sample.to_csv(OUT/'shared_sampling_indices.csv',index=False)
    crng=np.random.default_rng(np.random.SeedSequence(17,spawn_key=(1,)))
    calids=hierarchy_draw(frame,crng,3*512)
    cal=pd.DataFrame({'batch':np.repeat(np.arange(3),512),'slot':np.tile(np.arange(512),3),'cohort_index':calids})
    cal.to_csv(LOCAL/'calibration_indices.csv',index=False)
    scales=json.loads((V1/'moments.json').read_text())['scales']
    contract=dict(schema='pacld_crm_training_comparison_v2',baseline=BASE_SHA,checkpoint=str(CHECKPOINT),checkpoint_sha256=SHA,seed=17,groups={'B0':'FULL+Macro','B1':'FULL+Macro+lambda_normal*M_normal^2','B2':'FULL+Macro+lambda_future100*M_future100^2'},updates=64,effective_batch=512,microbatch=32,draws_per_group=32768,lambda_macro=1.,optimizer='Adam restore identical complete frozen optimizer state for all groups' if bundle.get('optimizer',{}).get('state') else 'fresh identical Adam for all groups',lr=bundle['config']['optimizer']['lr'],max_grad_norm=bundle['config']['training']['max_grad_norm'],precision=bundle['precision'],scales=scales,cohort=dict(windows=11221,flights=114,dates=17,support_s=.5,target='same native rate observation near .2s',training_distribution='CRM common-support .5s windows; NOT original .2/.5/1/2s mixed distribution'),sampling='seed17 IID date -> flight -> window, with replacement; shared order all groups; no adaptive probabilities',sampling_sha256=sha(OUT/'shared_sampling_indices.csv'),calibration_sha256=sha(LOCAL/'calibration_indices.csv'),calibration=dict(batches=3,effective_batch=512,independent_rng='SeedSequence(17,spawn_key=(1,)); independent draws, not necessarily disjoint identities',fit_batch=0,stability_batches=[1,2],target_gradient_ratio=.25,parameters='flow.dynamics.* (Slow, coupling, equilibrium, relaxation, Decoder)',stop_rules=dict(nonfinite=True,min_moment_to_base_norm=1e-8,max_lambda=1e4,weighted_ratio_range=[.0625,1.0],minimum_pairwise_cosine=0.),rule_origin='premeasurement operational stability gates; ratio must stay within factor four of 25%; any pairwise opposite Dynamics squared-moment direction stops; not performance acceptance thresholds'),moment='two pass, no-grad exact MB then backprop 2*MB*mean(m_i); all belief fields detached only on CRM branch',full_macro='original model(batch,1).total: window-mean FULL+Macro, invalid Macro=0; never divide by microbatch valid-pair counts',evaluation='only after all groups complete update64; initial plus B0/B1/B2; frozen PACLD156, A2 fixed feature identities, CIR306; no sealed test',publication='aggregate allowlist only; no new raw/window/checkpoint data published',status='prepared')
    write_json(OUT/'training_contract.json',contract)
    perupdate=sample.groupby('update').cohort_index.nunique()
    write_json(OUT/'sampling_summary.json',dict(draws=32768,unique_windows=sample.cohort_index.nunique(),repeated_draws=32768-sample.cohort_index.nunique(),dates=sample.date.nunique(),flights=sample.flight_id.nunique(),unique_per_update_min=int(perupdate.min()),unique_per_update_max=int(perupdate.max()),same_sequence_all_groups=True,calibration_draws=1536,calibration_unique=len(set(calids)),calibration_formal_identity_overlap=len(set(calids)&set(ids))))
    return contract

class Store:
    def __init__(self,bundle):
        self.bundle=bundle;self.frame=pd.read_csv(V1/'windows.csv');self.source=pd.read_pickle(SOURCE/'features.pkl')
        self.source=self.frame[['flight','t']].merge(self.source,on=['flight','t'],validate='one_to_one')
        assert len(self.source)==len(self.frame)
        self.cache=LOCAL/'windows';self.cache.mkdir(parents=True,exist_ok=True)
    def prepare(self,ids):
        missing=[i for i in sorted(set(map(int,ids))) if not (self.cache/f'{i}.pkl').exists()]
        if not missing:return
        # Sort by flight for the native reader LRU; the sampling order stays unchanged.
        ordered=self.frame.loc[missing].sort_values(['flight','t']).index.to_list()
        for start in range(0,len(ordered),128):
            ix=ordered[start:start+128];source=self.source.loc[ix]
            _,windows,queries=pickle.loads(prepare_batch((self.bundle['config'],'train',source)))
            for i,w,q in zip(ix,windows,queries):
                row=self.frame.loc[i];assert w.metadata['split']=='train'
                assert w.metadata['flight_id']==row.flight and abs(q[0]-row.query_s)<1e-10
                with (self.cache/f'{i}.pkl').open('wb') as f:pickle.dump((w,q[0]),f,protocol=5)
            print('cache',min(start+128,len(ordered)),'/',len(ordered),flush=True)
    @lru_cache(maxsize=1024)
    def get(self,i):
        with (self.cache/f'{int(i)}.pkl').open('rb') as f:return pickle.load(f)
    def chunks(self,ids,model,micro=32):
        for begin in range(0,len(ids),micro):
            ix=list(map(int,ids[begin:begin+micro]));entries=[self.get(i) for i in ix];ws=[e[0] for e in entries]
            batch=compile_windows(ws,model.config).to(next(model.parameters()).device,torch.float64)
            dt,u=pad_steps([steps(0,q,w.controls.canonical(),model.config.max_step) for w,q in entries])
            yield ix,batch,(dt.to(batch.initial.x),u.to(batch.initial.x))
    def moments(self,model,belief,plan,ids):
        b=belief.with_(**{k:getattr(belief,k).detach() for k in FIELDS})
        future=model.propagate_plan(b,plan);yhat=future.x[:,10]-b.x[:,10]
        rows=self.frame.loc[ids];sc=json.loads((V1/'moments.json').read_text())['scales']
        y=torch.as_tensor(rows.Y.to_numpy(),device=yhat.device)
        r=torch.as_tensor(rows[['R_normal','R_future100']].to_numpy()/np.array(sc['s_u'])[[0,2]],device=yhat.device)
        return (r*((y-yhat)/sc['s_p'])[:,None]).mean(0),yhat

@torch.no_grad()
def first_pass(model,store,ids,micro):
    M=np.zeros(2)
    for ix,batch,plan in store.chunks(ids,model,micro):
        m,_=store.moments(model,model.encode_batch(batch),plan,ix);M+=m.cpu().numpy()*len(ix)/len(ids)
    return M

def accumulated_backward(model,store,ids,micro,group,lam,M):
    model.zero_grad(set_to_none=True);values=np.zeros(3);valid=0;diagnostic=np.zeros(2)
    for ix,batch,plan in store.chunks(ids,model,micro):
        result=model(batch,1.);frac=len(ix)/len(ids);loss=result['total']
        if group=='B0':
            with torch.no_grad():m,_=store.moments(model,result['belief'],plan,ix)
        else:
            m,_=store.moments(model,result['belief'],plan,ix);j=0 if group=='B1' else 1
            loss=loss+lam*2*float(M[j])*m[j]
        diagnostic+=frac*m.detach().cpu().numpy()
        (frac*loss).backward()
        values+=frac*np.array([float(result['full'].mean().detach()),float(result['macro'].mean().detach()),float(result['total'].detach())]);valid+=int(result['macro_valid'].sum())
    return values,valid,diagnostic
