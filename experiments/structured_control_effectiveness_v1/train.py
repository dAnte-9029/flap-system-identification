"""One seed, three from-scratch architectures, equal six-epoch budget and loss."""
from common import *
import time, torch
from model import make_model
from system_identification.models.velocity_rate_gru import errors

def smoke(td,cfg,norm,device):
    ix=torch.arange(64,device=device);x=training.batch(td,ix);records=[]
    scales=torch.tensor(norm['loss_scales'],device=device);weights=torch.tensor(cfg['loss_weights'],device=device)
    for name in cfg['models']:
        training.set_seed(17);m=make_model(name,norm).to(device);opt=torch.optim.AdamW(m.parameters(),lr=.001)
        for step in range(3):
            opt.zero_grad();p=m(x);loss=training.window_loss(p,td['target'][ix],td['future_mask'][ix],scales,weights).mean()
            assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True)
            cg=float(m.effect_head.weight.grad.norm()) if name!='M0' else None
            if cg is not None:assert cg>0 and np.isfinite(cg)
            records.append(dict(model=name,step=step,loss=loss.item(),gradient_norm=grad.item(),control_gradient_norm=cg));opt.step()
    save(OUT/'smoke.json',dict(passed=True,records=records,horizon_steps=td['dt'].shape[1],training_restarts_from_scratch=True));print('H1000 smoke passed for all architectures',flush=True)

def run_model(name,td,vd,cfg,norm,device):
    folder=HERE/'training'/name;folder.mkdir(parents=True,exist_ok=False)
    training.set_seed(17);m=make_model(name,norm).to(device)
    opt=torch.optim.AdamW(m.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=6)
    scales=torch.tensor(norm['loss_scales'],device=device);weights=torch.tensor(cfg['loss_weights'],device=device)
    meta=pd.read_parquet(TRAIN/'validation_windows.parquet');fi=torch.tensor(pd.factorize(meta.flight_id)[0],device=device)
    best=float('inf');updates=0;records=[];start=time.time();torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1,7):
        tick=time.time();m.train();gen=torch.Generator().manual_seed(17*10000+epoch)
        cpu=torch.randperm(len(td['initial']),generator=gen);orderhash=hashlib.sha256(cpu.numpy().tobytes()).hexdigest();order=cpu.to(device)
        total=0.;grads=[];controlgrads=[]
        for ix in order.split(cfg['batch_size']):
            opt.zero_grad(set_to_none=True);pred=m(training.batch(td,ix))
            loss=training.window_loss(pred,td['target'][ix],td['future_mask'][ix],scales,weights).mean()
            if not torch.isfinite(loss):raise RuntimeError(f'{name}: nonfinite loss')
            loss.backward()
            if name!='M0':
                cg=m.effect_head.weight.grad.norm().item();assert np.isfinite(cg) and cg>0;controlgrads.append(cg)
            gn=torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);opt.step()
            grads.append(gn.item());total+=loss.item()*len(ix);updates+=1
        vl,wvl=training.validate(m,vd,scales,weights,fi,cfg['evaluation_batch_size'])
        assert np.isfinite(vl)
        checkpoint=dict(model=m.state_dict(),normalization=norm,config=cfg,model_name=name,seed=17,epoch=epoch,validation_flight_loss=vl,input_keys=training.INPUT_KEYS)
        if vl<best-cfg['minimum_improvement']:
            best=vl;best_epoch=epoch;torch.save(checkpoint,folder/'best.pt')
        torch.save(checkpoint,folder/'last.pt')
        torch.cuda.synchronize(device)
        record=dict(model=name,epoch=epoch,updates=updates,train_window_loss=total/len(order),validation_flight_loss=vl,validation_window_loss=wvl,lr=opt.param_groups[0]['lr'],gradient_norm_mean=float(np.mean(grads)),gradient_norm_max=max(grads),control_gradient_norm_mean=float(np.mean(controlgrads)) if controlgrads else None,seconds=time.time()-tick,batch_order_sha256=orderhash,best_epoch=best_epoch,peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        records.append(record)
        with (folder/'training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True);sched.step()
    save(folder/'summary.json',dict(model=name,seed=17,epochs=6,updates=updates,examples_seen=6*len(td['initial']),parameters=sum(p.numel() for p in m.parameters()),best_epoch=best_epoch,best_validation_flight_loss=best,seconds=time.time()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(device),best_sha256=sha(folder/'best.pt'),last_sha256=sha(folder/'last.pt'),stop_reason='fixed_epoch_budget'))
    return records

def main():
    cfg=json.loads((HERE/'config.json').read_text());norm=json.loads((HERE/'normalization.json').read_text());r=json.loads((OUT/'source_receipt.json').read_text())
    assert sha(HERE/'config.json')==r['config_sha256']
    assert 'passed' in (HERE/'tests.log').read_text()
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=True
    device=torch.device(cfg['device']);torch.cuda.set_device(device)
    start=time.time();td=load_data('train',device);vd=load_data('validation',device)
    save(HERE/'runtime.json',dict(python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,device=str(device),gpu=torch.cuda.get_device_name(device),load_seconds=time.time()-start,tf32=True,cudnn_deterministic=True,test_read=False,training_source_sha256={p.name:sha(p) for p in [HERE/'model.py',Path(__file__),HERE/'common.py']},data_bytes=sum(v.numel()*v.element_size() for d in [td,vd] for v in d.values())))
    smoke(td,cfg,norm,device);records=[]
    for name in cfg['models']:records.extend(run_model(name,td,vd,cfg,norm,device))
    pd.DataFrame(records).to_csv(OUT/'training_history.csv',index=False)
    assert all(g.batch_order_sha256.nunique()==1 for _,g in pd.DataFrame(records).groupby('epoch'))
    print('Completed exactly 3 x 6 epochs; identical batch order; no other training',flush=True)
if __name__=='__main__':main()
