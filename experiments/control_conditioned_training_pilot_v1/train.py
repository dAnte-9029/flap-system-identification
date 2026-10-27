"""Frozen 12-run W0/W1 pilot. Exact window-weight alignment and epoch resume state."""
from common import *
import argparse,os,time,random,torch
def atomic_checkpoint(path,value):
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);os.replace(tmp,path)

def run_one(fold,seed,name,td,vd,device,cfg,norm,ids):
    folder=HERE/'folds'/f'fold{fold}'/'training'/name/f'seed{seed}'
    if (folder/'summary.json').exists():
        done=json.loads((folder/'summary.json').read_text());assert done['epochs']==6 and sha(folder/'best.pt')==done['best_sha256'];return
    folder.mkdir(parents=True,exist_ok=True);training.set_seed(seed);m=make_model(name,norm).to(device);initial_hash=hashlib.sha256(b''.join(k.encode()+v.detach().cpu().numpy().tobytes() for k,v in sorted(m.state_dict().items()))).hexdigest();scheme=name[-2:]
    opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=6)
    fi=torch.tensor(pd.factorize(pd.read_parquet(TRAIN/'validation_windows.parquet').flight_id)[0],device=device)
    scales=torch.tensor(norm['loss_scales'],device=device);weights=torch.tensor(cfg['loss_weights'],device=device)
    best=float('inf');best_epoch=0;updates=0;records=[];start_epoch=1;resumed=False
    if (folder/'last.pt').exists():
        ck=torch.load(folder/'last.pt',map_location=device,weights_only=False)
        assert ck['config']==cfg and ck['normalization']==norm and ck['source_sha256']==sha(PILOT/'model.py')
        assert ck['trainer_sha256']==sha(Path(__file__)) and ck['weight_file_sha256']==sha(WEIGHTS/f'results/fold{fold}/weights.npz')
        m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler'])
        records=ck['records'];updates=ck['updates'];best=ck['best_loss'];best_epoch=ck['best_epoch'];start_epoch=ck['epoch']+1
        torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state(ck['cuda_rng'].cpu(),device);random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);resumed=True
        (folder/'training.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    torch.cuda.reset_peak_memory_stats(device)
    provenance=dict(scheme=scheme,initial_model_sha256=initial_hash,weight_file_sha256=sha(WEIGHTS/f'results/fold{fold}/weights.npz'),weight_loss_source_sha256=sha(WEIGHTS/'weighted_loss.py'),fold=fold,seed=seed,model=name,config_sha256=sha(HERE/'config.json'),normalization_sha256=sha(HERE/'folds'/f'fold{fold}'/'normalization.json'),train_indices_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),model_source_sha256=sha(PILOT/'model.py'),trainer_sha256=sha(Path(__file__)),common_sha256=sha(HERE/'common.py'),device=str(device),gpu=torch.cuda.get_device_name(device),torch=torch.__version__,cuda=torch.version.cuda,train_windows=len(td['initial']),validation_windows=len(vd['initial']),test_read=False)
    if not (folder/'provenance.json').exists():save(folder/'provenance.json',provenance)
    for epoch in range(start_epoch,7):
        tick=time.time();m.train();cpu=torch.randperm(len(td['initial']),generator=torch.Generator().manual_seed(seed*10000+epoch));order=cpu.to(device)
        order_hash=hashlib.sha256(ids[cpu.numpy()].tobytes()).hexdigest();total=0.;uniform_total=0.;grads=[];cg=[];batch_stats=[]
        for ix in order.split(1024):
            opt.zero_grad(set_to_none=True);p=m(training.batch(td,ix));bw=td['window_weight'][ix] if scheme=='W1' else torch.ones(len(ix),dtype=td['window_weight'].dtype,device=device)
            loss=weighted.weighted_loss(p,td['target'][ix],td['future_mask'][ix],scales,weights,scheme=scheme,window_weight=bw)
            with torch.no_grad():uniform=training.window_loss(p,td['target'][ix],td['future_mask'][ix],scales,weights).mean()
            uniform_total+=uniform.item()*len(ix)
            bs=bw.sum().item();bss=bw.square().sum().item();batch_stats.append(dict(batch=len(batch_stats),windows=len(ix),weight_mean=bs/len(ix),weight_sum=bs,weight_ess=bs*bs/bss,weight_ess_ratio=bs*bs/bss/len(ix)))
            if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
            loss.backward()
            if name.startswith('M1'):
                value=float(m.effect_head.weight.grad.norm());assert value>0 and np.isfinite(value);cg.append(value)
            grad=torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);opt.step();updates+=1;grads.append(float(grad));total+=loss.item()*len(ix)
        vl,wvl=training.validate(m,vd,scales,weights,fi,2048);assert np.isfinite(vl)
        ck=dict(model=m.state_dict(),normalization=norm,config=cfg,seed=seed,fold=fold,model_name=name,epoch=epoch,validation_flight_loss=vl,input_keys=training.INPUT_KEYS,source_sha256=sha(PILOT/'model.py'))
        if vl<best-cfg['minimum_improvement']:
            best=vl;best_epoch=epoch;atomic_checkpoint(folder/'best.pt',ck)
        torch.cuda.synchronize(device)
        record=dict(scheme=scheme,train_uniform_window_loss=uniform_total/len(order),batch_weight_mean=float(np.mean([b['weight_mean'] for b in batch_stats])),batch_weight_sum=float(sum(b['weight_sum'] for b in batch_stats)),batch_ess_mean=float(np.mean([b['weight_ess'] for b in batch_stats])),batch_ess_ratio_min=float(min(b['weight_ess_ratio'] for b in batch_stats)),batch_weight_statistics=batch_stats,fold=fold,model=name,seed=seed,epoch=epoch,updates=updates,train_window_loss=total/len(order),validation_flight_loss=vl,validation_window_loss=wvl,lr=opt.param_groups[0]['lr'],gradient_norm_mean=float(np.mean(grads)),gradient_norm_max=max(grads),control_gradient_norm_mean=float(np.mean(cg)) if cg else None,control_gradient_norm_min=min(cg) if cg else None,seconds=time.time()-tick,batch_order_sha256=order_hash,best_epoch=best_epoch,peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        records.append(record);sched.step()
        ck.update(trainer_sha256=sha(Path(__file__)),weight_file_sha256=sha(WEIGHTS/f'results/fold{fold}/weights.npz'),optimizer=opt.state_dict(),scheduler=sched.state_dict(),records=records,updates=updates,best_loss=best,best_epoch=best_epoch,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device),python_rng=random.getstate(),numpy_rng=np.random.get_state())
        atomic_checkpoint(folder/'last.pt',ck)
        with (folder/'training.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        print(json.dumps({k:v for k,v in record.items() if k!='batch_weight_statistics'}),flush=True)
    save(folder/'summary.json',dict(fold=fold,model=name,seed=seed,epochs=6,updates=updates,examples_seen=6*len(td['initial']),parameters=sum(p.numel() for p in m.parameters()),best_epoch=best_epoch,best_validation_flight_loss=best,seconds=sum(r['seconds'] for r in records),peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in records),best_sha256=sha(folder/'best.pt'),last_sha256=sha(folder/'last.pt'),resumed=resumed,stop_reason='fixed_6_epoch_budget',**{k:v for k,v in provenance.items() if k not in ('fold','model','seed')}))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda:1');args=parser.parse_args()
    cfg=json.loads((HERE/'config.json').read_text());receipt=json.loads((OUT/'source_receipt.json').read_text());assert receipt['config_sha256']==sha(HERE/'config.json')
    assert json.loads((OUT/'pretraining_tests.json').read_text())['passed']
    device=setup(args.device);free,total=torch.cuda.mem_get_info(device);assert free>4*1024**3,(free,total)
    raw=load_numpy('train');vd=to_device(load_numpy('validation'),device);indices=np.load(OUT/'holdout_window_indices.npz')
    for fold in [0,1,2]:
        ids=indices[f'fold{fold}_train_indices'];norm=json.loads((HERE/'folds'/f'fold{fold}'/'normalization.json').read_text())
        raw_weights=np.load(WEIGHTS/f'results/fold{fold}/weights.npz');assert np.array_equal(raw_weights['registered_index'],ids)
        assert sha(WEIGHTS/f'results/fold{fold}/weights.npz')==json.loads((HERE/'folds'/f'fold{fold}'/'weight_alignment.json').read_text())['weights_sha256']
        td=to_device(raw,device,ids);td['window_weight']=torch.from_numpy(raw_weights['w1']).to(device)
        try:
            for name in COMBOS:run_one(fold,17,name,td,vd,device,cfg,norm,ids)
        except Exception as exc:
            save(OUT/'training_failure.json',dict(fold=fold,error=repr(exc),time=time.time()));raise
        finally:del td;torch.cuda.empty_cache()
    print('All 12 runs complete',flush=True)
if __name__=='__main__':main()
