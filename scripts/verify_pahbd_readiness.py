#!/usr/bin/env python3
"""Coverage plus FOUR real train forward/backward checks. No optimizer/training run."""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from system_identification.models.pahbd import PAHBD,Config
from system_identification.data.pahbd import NativeWindows,configure_latent
from system_identification.data.pahbd_synthetic import controlled_periodic_window
from system_identification.training.pahbd import window_loss,FlightBalancedSampler
from system_identification.evaluation.pahbd import phase_scenarios,FrozenRollResponse


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--coverage-only',action='store_true');args=p.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(Path('configs/pahbd/default.json').read_text());torch.set_num_threads(1);torch.manual_seed(cfg['seed'])
    model=PAHBD(Config(**cfg['model'])).double();coverage=[];train=None
    for split in ('train','validation'):
        data=NativeWindows(**cfg['data'],split=split)
        frame=data.windows
        groups=frame.groupby(['flight_id','horizon_us']).agg(windows=('window_id','size'),relative=('relative_phase_eligible','sum'),shared=('shared_periodic_eligible','sum'),reference_changed=('reference_updates',lambda x:int((x>0).sum())),hall_traceable=('hall_event_traceable','sum')).reset_index()
        groups.to_csv(output/f'{split}_coverage_per_flight.csv',index=False)
        for h,g in frame.groupby('horizon_us'):
            coverage.append(dict(split=split,horizon_us=int(h),flights=int(g.flight_id.nunique()),average_windows=len(g),relative_windows=int(g.relative_phase_eligible.sum()),shared_windows=int(g.shared_periodic_eligible.sum()),hall_traceable_windows=int(g.hall_event_traceable.sum()),software_reference_update_windows=int((g.reference_updates>0).sum()),actual_horizon_s_min=float((g.last_target_measurement_us-g.decision_us).min()*1e-6),actual_horizon_s_max=float((g.last_target_measurement_us-g.decision_us).max()*1e-6)))
        if split=='train':train=data
    (output/'coverage.json').write_text(json.dumps(coverage,indent=2))
    sampler=FlightBalancedSampler(train.windows,cfg['seed'])
    for _ in range(4*len(sampler.groups)):sampler.draw()
    (output/'sampler_coverage.json').write_text(json.dumps(sampler.coverage(),indent=2))
    if args.coverage_only:return
    records=[];used=set()
    for horizon in (200000,500000,1000000,2000000):
        eligible=train.windows[(train.windows.horizon_us==horizon)&train.windows.relative_phase_eligible&~train.windows.flight_id.isin(used)]
        index=int(eligible.index[0]);t=time.perf_counter();w=configure_latent(train[index],model.config);load=time.perf_counter()-t
        used.add(w.metadata['flight_id']);model.zero_grad(set_to_none=True)
        t=time.perf_counter();loss,parts,belief,states,times=window_loss(model,w,**cfg['loss']);forward=time.perf_counter()-t
        t=time.perf_counter();loss.backward();backward=time.perf_counter()-t
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        if not torch.isfinite(loss) or not grads or not all(torch.isfinite(g).all() for g in grads):raise RuntimeError('invalid real-window backward')
        if horizon>=500000 and not w.metadata['cycle_pairs']:raise RuntimeError('empty cross-wingbeat supervision')
        record=dict(flight_id=w.metadata['flight_id'],window_id=w.metadata['window_id'],nominal_horizon_us=horizon,actual_horizon_s=w.metadata['actual_horizon_s'],load_seconds=load,forward_seconds=forward,backward_seconds=backward,process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,history=len(w.history),targets=len(w.targets),controls=len(w.controls.times),cycle_pairs=len(w.metadata['cycle_pairs']),periodic_enabled=w.metadata['periodic_enabled'],loss_finite=True,gradients_finite=True,quaternion_norm_max_error=max(float((s.x[6:10].norm()-1).abs().detach()) for s in states),joint_cov_min_eigenvalue=min(float(torch.linalg.eigvalsh(s.phase_covariance).min().detach()) for s in states),joint_cov_f_phi=float(states[-1].phase_covariance[0,1].detach()),replay=model.belief.last_replay)
        records.append(record);print(json.dumps(record),flush=True)
        (output/'real_train_checks.json').write_text(json.dumps(records,indent=2))
        del loss,parts,belief,states,w
    # Long-horizon scenario mechanism, synthetic and no optimizer, same production flow.
    small=PAHBD(Config(width=16,latent=4)).double();w,_=controlled_periodic_window(small.config,2.)
    t=time.perf_counter()
    with torch.no_grad():
        belief=small.encode(w);scenarios=phase_scenarios(small,belief,[.2,.5,1.,2.],w.controls)
        phase_mean=torch.stack([a[-1].phi for a in scenarios['states']]).mean(0)
        error=float((phase_mean-scenarios['mean_phase'][-1].phi).abs().max())
    if error>1e-10:raise RuntimeError('scenario phase mean mismatch')
    source=Path('/home/zn/flap-system-identification/.worktrees/local-control-response-rollout-v1/experiments/local_control_response_rollout_v1')
    frozen=FrozenRollResponse(source)
    result=dict(code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),seed=cfg['seed'],device='cpu',dtype='float64',threads=1,model=cfg['model'],formal_optimizer_steps=0,coverage=coverage,real_train_checks=records,scenario_2s=dict(count=4,phase_mean_error_rad=error,seconds=time.perf_counter()-t,process_noise_calibrated=False),control_cohort=dict(train_pairs=int((frozen.pairs.split=='train').sum()),train_components=int(frozen.pairs.loc[frozen.pairs.split=='train','dependency_component'].nunique()),source=str(source)))
    (output/'verification.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
