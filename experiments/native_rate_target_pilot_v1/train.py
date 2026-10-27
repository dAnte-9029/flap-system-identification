"""Six fixed-budget runs, last epoch only; original inputs and original loss."""
from common import *
import argparse,time,os,random,torch

def atomic(path,value):
 tmp=path.with_suffix('.tmp');torch.save(value,tmp);os.replace(tmp,path)

def statehash(m):return hashlib.sha256(b''.join(k.encode()+v.detach().cpu().numpy().tobytes() for k,v in sorted(m.state_dict().items()))).hexdigest()

@torch.inference_mode()
def validate_both(m,d,scales,weights,fi):
 m.eval();values={'A':[],'B':[]}
 for ix in torch.arange(len(d['initial']),device=scales.device).split(2048):
  p=m(training.batch(d,ix))
  for label in values:values[label].append(training.window_loss(p,d['target_'+label][ix],d['loss_mask'][ix],scales,weights))
 out={}
 for label,ls in values.items():
  v=torch.cat(ls);out['validation_'+label+'_flight_loss']=(torch.bincount(fi,weights=v)/torch.bincount(fi)).mean().item();out['validation_'+label+'_window_loss']=v.mean().item()
 return out

def run_one(fold,name,td,vd,device,cfg,norm,ids):
 folder=HERE/'folds'/f'fold{fold}'/'training'/name/'seed17';folder.mkdir(parents=True,exist_ok=True)
 if (folder/'summary.json').exists():
  done=json.loads((folder/'summary.json').read_text());assert done['epochs']==6 and done['last_sha256']==sha(folder/'last.pt');return
 training.set_seed(17);m=make_model(name,norm).to(device);initial_hash=statehash(m);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=6)
 scales=torch.tensor(norm['loss_scales'],device=device);weights=torch.ones(4,device=device);fi=torch.tensor(pd.factorize(pd.read_parquet(TRAIN/'validation_windows.parquet').flight_id)[0],device=device);records=[];updates=0;start=1
 provenance=dict(fold=fold,model=name,seed=17,initial_model_sha256=initial_hash,normalization_sha256=sha(HERE/'folds'/f'fold{fold}'/'normalization.json'),train_indices_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),config_sha256=sha(HERE/'config.json'),trainer_sha256=sha(Path(__file__)),model_sha256=sha(ROOT/'src/system_identification/models/velocity_rate_gru.py'),target_pack_sha256=sha(OUT/'train_targets.npz'),gpu=torch.cuda.get_device_name(device),device=str(device),torch=torch.__version__,cuda=torch.version.cuda,parameters=sum(p.numel() for p in m.parameters()),test_read=False,checkpoint_selection='fixed epoch 6')
 if (folder/'last.pt').exists():
  ck=torch.load(folder/'last.pt',map_location=device,weights_only=False);assert ck['provenance']==provenance and ck['config']==cfg and ck['normalization']==norm
  m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);records=ck['records'];updates=ck['updates'];start=ck['epoch']+1
  torch.set_rng_state(ck['torch_rng'].cpu());torch.cuda.set_rng_state(ck['cuda_rng'].cpu(),device);np.random.set_state(ck['numpy_rng']);random.setstate(ck['python_rng'])
 save(folder/'provenance.json',provenance);torch.cuda.reset_peak_memory_stats(device)
 for epoch in range(start,7):
  tick=time.time();m.train();cpu=torch.randperm(len(ids),generator=torch.Generator().manual_seed(170000+epoch));order=cpu.to(device);total=0.;grads=[]
  for ix in order.split(1024):
   opt.zero_grad(set_to_none=True);p=m(training.batch(td,ix));loss=training.window_loss(p,td['target_'+name[-1]][ix],td['loss_mask'][ix],scales,weights).mean()
   assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);opt.step();updates+=1;grads.append(float(grad));total+=loss.item()*len(ix)
  validation=validate_both(m,vd,scales,weights,fi);assert all(np.isfinite(v) for v in validation.values());torch.cuda.synchronize(device)
  record=dict(fold=fold,model=name,seed=17,epoch=epoch,updates=updates,train_window_loss=total/len(ids),**validation,lr=opt.param_groups[0]['lr'],gradient_norm_mean=float(np.mean(grads)),gradient_norm_max=max(grads),seconds=time.time()-tick,batch_order_sha256=hashlib.sha256(ids[cpu.numpy()].tobytes()).hexdigest(),peak_allocated_bytes=torch.cuda.max_memory_allocated(device));records.append(record);scheduler.step()
  ck=dict(model=m.state_dict(),normalization=norm,config=cfg,epoch=epoch,updates=updates,records=records,provenance=provenance,optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device),numpy_rng=np.random.get_state(),python_rng=random.getstate())
  atomic(folder/'last.pt',ck);(folder/'training.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records));print(json.dumps(record),flush=True)
 save(folder/'summary.json',dict(**provenance,epochs=6,updates=updates,seconds=sum(r['seconds'] for r in records),peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in records),last_sha256=sha(folder/'last.pt'),train_windows=len(ids),train_examples=6*len(ids),stop_reason='fixed budget'))

def main():
 pa=argparse.ArgumentParser();pa.add_argument('--device',default='cuda:1');args=pa.parse_args();assert json.loads((OUT/'pretraining_tests.json').read_text())['passed'];cfg=json.loads((HERE/'config.json').read_text());device=setup(args.device);assert torch.cuda.mem_get_info(device)[0]>4*1024**3
 raw=with_targets(load_numpy('train'),target_pack('train'));vd=to_device(with_targets(load_numpy('validation'),target_pack('validation')),device);indices=np.load(OUT/'holdout_window_indices.npz')
 for fold in range(3):
  ids=indices[f'fold{fold}_train_indices'];norm=json.loads((HERE/'folds'/f'fold{fold}'/'normalization.json').read_text());td=to_device(raw,device,ids)
  for name in COMBOS:run_one(fold,name,td,vd,device,cfg,norm,ids)
  del td;torch.cuda.empty_cache()
 print('6/6 runs complete',flush=True)
if __name__=='__main__':main()
