#!/usr/bin/env python3
"""Run the frozen September local/reset versus free-run baseline experiment."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from system_identification.data.september_trajectory import file_hash, verify_registered_dataset
from system_identification.evaluation.trajectory_rollout_diagnostics import (
    ERROR_NAMES, error_curve, origin_metadata, prediction_errors, require_history, shifted_local_windows,
)
from system_identification.models.trajectory import ConstantTwistPredictor, IntegratedDynamicsPredictor
from system_identification.training.trajectory_baselines import fit_ridge_dynamics
from system_identification.training.trajectory_main_v1 import (
    MainV1Config, assemble_history_trajectory_windows, fit_history_trajectory_model,
    fit_main_v1_stats, predict_history_trajectory_model,
)


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data,indent=2,default=str)+"\n")
    temporary.replace(path)


def status(output, state, **extra):
    write_json(output/"status.json", {"state":state,"pid":os.getpid(),
        "updated_at":datetime.now(timezone.utc).isoformat(),**extra})
    print(state,extra,flush=True)


def prepare(registry, train_horizon=1):
    manifest=verify_registered_dataset(registry,ROOT)
    if manifest["dataset_version"]!="trajectory_dataset_v2_september":
        raise ValueError("requires September v2 dataset")
    if manifest["split_contract"]["materialized_partitions"]!=["train","validation"]:
        raise ValueError("requires unopened-test train/validation dataset")
    reg=yaml.safe_load(registry.read_text())
    entry=reg["datasets"][reg["default_dataset_id"]]
    data_root=(ROOT/entry["manifest_path"]).parent
    samples={}; windows={}; coverage=[]
    for part,horizon in [("train",train_horizon),("validation",5)]:
        info=manifest["partitions"][part]
        samples[part]=pd.read_parquet(data_root/info["samples_file"])
        original=pd.read_parquet(data_root/info["cohorts"]["core"]["windows"][str(horizon)]["file"])
        windows[part]=require_history(original)
        if set(windows[part].log_id)!=set(manifest["split_contract"]["assignments"][part]):
            raise ValueError("history filtering dropped a full log")
        lookup=samples[part].set_index(["log_id","sample_in_log"])
        for r in windows[part].itertuples():
            start=lookup.loc[(r.log_id,r.start_sample_in_log)]
            end=lookup.loc[(r.log_id,r.end_sample_in_log)]
            if (start.timestamp_us!=r.start_timestamp_us or end.timestamp_us!=r.end_timestamp_us
                    or start.segment_id!=r.segment_id or end.segment_id!=r.segment_id
                    or start.sample_in_segment!=r.start_sample_in_segment):
                raise ValueError("core window source identity mismatch")
        for log, group in original.groupby("log_id"):
            coverage.append({"partition":part,"log_id":log,"horizon_s":horizon,
                "original_windows":len(group),"full_history_windows":int((windows[part].log_id==log).sum())})
    return manifest,entry,samples,windows,pd.DataFrame(coverage)


def summarize_curves(curves):
    numeric=[c for c in curves if "rmse" in c or c.endswith("p95") or c.endswith("fraction")]
    keys=["model","mode","local_horizon_s","global_horizon_s"]
    result=curves.groupby(keys,sort=False)[numeric].agg(lambda x:float(np.mean(x.to_numpy()))).reset_index()
    return result


def evaluate(model_name, predictor, samples, parents, batch, output, mode, offset):
    if hasattr(predictor,"recurrent_cell"):
        prediction=predict_history_trajectory_model(predictor,batch,use_history=True,batch_size=256,device="cuda:1")
    else:
        prediction=predictor.rollout(batch.trajectory.initial_state(),batch.trajectory.controls,batch.trajectory.dt_s)
    truth=batch.trajectory.truth
    errors=prediction_errors(prediction,truth)
    meta=origin_metadata(samples,parents)
    curve=error_curve(errors,batch.trajectory.log_ids,batch.trajectory.dt_s,model_name,mode,offset)
    # Persist all error samples, not only selected endpoints. Prediction examples preserve all states.
    label=f"{model_name}_{mode}_{offset}"
    np.savez_compressed(output/"errors"/f"{label}.npz",**errors,
        window_ids=batch.trajectory.window_ids.astype(str),log_ids=batch.trajectory.log_ids.astype(str),
        dt_s=batch.trajectory.dt_s)
    example=np.array([np.flatnonzero(batch.trajectory.log_ids==log)[0] for log in np.unique(batch.trajectory.log_ids)])
    values={f"prediction_{k}":v[example] for k,v in vars(prediction).items()}
    values.update({f"truth_{k}":v[example] for k,v in vars(truth).items()})
    np.savez_compressed(output/"examples"/f"{label}.npz",**values,
        window_ids=batch.trajectory.window_ids[example].astype(str),dt_s=batch.trajectory.dt_s[example])
    endpoints=[]
    steps=[50,100,150,250] if mode=="continuous" else [50]
    for step in steps:
        frame=meta.copy()
        frame["model"]=model_name; frame["mode"]=mode
        frame["local_horizon_s"]=step/50; frame["global_horizon_s"]=offset+step/50
        if "parent_window_id" in parents:
            frame["parent_window_id"]=parents.parent_window_id.to_numpy()
        else:
            frame["parent_window_id"]=parents.window_id.to_numpy()
        frame["observed_local_horizon_s"]=batch.trajectory.dt_s[:,:step].sum(1)
        for name,value in errors.items():
            frame[name]=value[:,step]
            frame[name+"_trajectory_rms"]=np.sqrt(np.mean(value[:,1:step+1]**2,axis=1))
        frame["nonfinite"]=~np.isfinite(frame[list(ERROR_NAMES)]).all(axis=1)
        frame["large_error"]=(frame.position_error_m>10)|(frame.attitude_error_deg>60)|frame.nonfinite
        endpoints.append(frame)
    curve.to_csv(output/"curves"/f"{label}.csv",index=False)
    return curve,pd.concat(endpoints,ignore_index=True)


def plot(curve, output):
    fig, axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    for axis,name in zip(axes.flat,ERROR_NAMES):
        metric=name.replace("error","rmse")
        for (model,mode),g in curve.groupby(["model","mode"]):
            g=g.sort_values("global_horizon_s")
            axis.plot(g.global_horizon_s,g[metric],label=f"{model}/{mode}",
                      linestyle="-" if mode=="continuous" else "--",alpha=.8)
        axis.set_xlabel("Time from parent origin (s)");axis.set_ylabel(metric);axis.grid(alpha=.25)
    axes[0,0].legend(fontsize=6)
    fig.savefig(output/"rollout_error_curves.png",dpi=160);plt.close(fig)


def run(args, output):
    if not torch.cuda.is_available() or torch.cuda.device_count()<2:
        raise RuntimeError("GPU 1 unavailable; no CPU fallback")
    torch.set_num_threads(4);torch.manual_seed(17)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    status(output,"loading_and_verifying_data")
    manifest,entry,samples,windows,coverage=prepare(args.registry)
    coverage.to_csv(output/"coverage.csv",index=False)
    for part,w in windows.items():w.to_parquet(output/f"{part}_selected_windows.parquet",index=False)
    config_common=dict(use_history=True,objective_steps=50,hidden_size=64,epochs=40,
        batch_size=256,learning_rate=3e-4,weight_decay=1e-5,gradient_clip_norm=5,seed=17)
    configs=[MainV1Config(model_name="history_no_control",use_controls=False,**config_common),
             MainV1Config(model_name="history_controlled",use_controls=True,**config_common)]
    sources=[Path(__file__),ROOT/"src/system_identification/evaluation/trajectory_rollout_diagnostics.py",
        ROOT/"src/system_identification/models/trajectory_main_v1.py",
        ROOT/"src/system_identification/models/trajectory.py",
        ROOT/"src/system_identification/training/trajectory_main_v1.py",
        ROOT/"src/system_identification/training/trajectory_baselines.py"]
    run_manifest={"experiment":"september_step2_baselines","dataset":entry,"dataset_id":manifest["dataset_id"],
        "split_contract":manifest["split_contract"],"sampling":manifest["sampling"],
        "phase_contract":"original GRU relative-to-origin phase; Hall/logged phase used only for grouping",
        "future_known":manifest["roles"]["known_future_control_t0_to_tT_exclusive"],
        "future_forbidden":manifest["roles"]["future_forbidden_as_input"],
        "training_configs":[asdict(c) for c in configs],"ridge_alpha":1,"device":"cuda:1",
        "python_executable":sys.executable,"source_hashes":{str(p.relative_to(ROOT)):file_hash(p) for p in sources},
        "git_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "git_status":subprocess.check_output(["git","status","--short"],cwd=ROOT,text=True),
        "protocol":"docs/contracts/2026-09-10_september_step2_baselines.md",
        "validation_fitting_or_early_stopping":False,"sealed_test_opened":False,
        "models":["constant_twist","ridge_no_control",*[c.model_name for c in configs]],
        "large_error_thresholds":{"position_m":10,"attitude_deg":60},
        "train_windows":len(windows["train"]),"validation_parent_windows":len(windows["validation"])}
    write_json(output/"manifest.json",run_manifest)
    train=assemble_history_trajectory_windows(samples["train"],windows["train"],history_steps=26)
    if not train.history_mask.all():raise ValueError("unexpected padded training history")
    stats=fit_main_v1_stats(samples["train"],train)
    models={"constant_twist":ConstantTwistPredictor()}
    ridge,_=fit_ridge_dynamics(samples["train"],alpha=1,use_controls=False)
    np.savez(output/"models"/"ridge_no_control.npz",**vars(ridge))
    models["ridge_no_control"]=IntegratedDynamicsPredictor(ridge)
    for config in configs:
        status(output,"training",model=config.model_name)
        model,history=fit_history_trajectory_model(train,stats,config,device="cuda:1")
        torch.save({"state_dict":model.state_dict(),"config":asdict(config),"history_steps":26},
                   output/"models"/f"{config.model_name}.pt")
        history.to_csv(output/f"training_history_{config.model_name}.csv",index=False)
        models[config.model_name]=model
    del train
    all_curves=[];all_endpoints=[]
    parents=windows["validation"]
    for mode,offset in [("continuous",0),*[("reset_1s",i) for i in range(5)]]:
        selected=parents if mode=="continuous" else shifted_local_windows(samples["validation"],parents,offset*50)
        batch=assemble_history_trajectory_windows(samples["validation"],selected,history_steps=26)
        if not batch.history_mask.all():raise ValueError("unexpected padded validation history")
        for name,predictor in models.items():
            status(output,"evaluating",model=name,mode=mode,offset_s=offset)
            curves,endpoints=evaluate(name,predictor,samples["validation"],selected,batch,output,mode,offset)
            all_curves.append(curves);all_endpoints.append(endpoints)
        del batch
    curves=pd.concat(all_curves,ignore_index=True); endpoints=pd.concat(all_endpoints,ignore_index=True)
    curves.to_csv(output/"per_log_error_curves.csv",index=False)
    macro=summarize_curves(curves);macro.to_csv(output/"equal_log_error_curves.csv",index=False)
    macro.loc[macro.global_horizon_s.isin([1,2,3,5])].to_csv(output/"endpoint_metrics.csv",index=False)
    endpoints.to_parquet(output/"window_endpoint_metrics.parquet",index=False)
    for group in [None,"maneuver_at_origin","logged_phase_bin_at_origin"]:
        keys=["model","mode","global_horizon_s","log_id"]+([group] if group else [])
        values=list(ERROR_NAMES)+[n+"_trajectory_rms" for n in ERROR_NAMES]
        perlog=endpoints.groupby(keys)[values].agg(lambda x:float(np.sqrt(np.mean(x.to_numpy()**2)))).reset_index()
        counts=endpoints.groupby(keys).size().rename("n_windows").reset_index()
        perlog=perlog.merge(counts,on=keys,validate="one_to_one")
        perlog.to_csv(output/f"per_log_{group or 'overall'}_metrics.csv",index=False)
        macro_keys=[k for k in keys if k!="log_id"]
        grouped=perlog.groupby(macro_keys)[values].agg(lambda x:float(np.mean(x.to_numpy()))).reset_index()
        group_counts=perlog.groupby(macro_keys).agg(n_logs=("log_id","nunique"),n_windows=("n_windows","sum")).reset_index()
        grouped=grouped.merge(group_counts,on=macro_keys,validate="one_to_one")
        grouped.to_csv(output/f"macro_{group or 'overall'}_metrics.csv",index=False)
    plot(macro,output)
    write_json(output/"summary.json",{"experiment":run_manifest["experiment"],"status":"completed",
        "models":list(models),"train_windows":len(windows["train"]),"validation_parent_windows":len(parents),
        "interpretation":"single-seed validation diagnostic; no scientific promotion decision",
        "sealed_test_opened":False,"artifacts":[p.name for p in output.iterdir()]})
    status(output,"completed")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry",type=Path,default=ROOT/"configs/data/trajectory_dataset_registry.yaml")
    parser.add_argument("--output-root",type=Path,default=ROOT/"artifacts/september_step2_20260910")
    parser.add_argument("--preflight",action="store_true")
    args=parser.parse_args()
    if args.preflight:
        _,_,_,windows,coverage=prepare(args.registry)
        torch.ones(1,device="cuda:1")
        print(coverage.to_string(index=False));print({p:len(w) for p,w in windows.items()})
        return
    output=args.output_root.resolve()
    output.mkdir(parents=True,exist_ok=False)
    try:
        for directory in ["models","errors","examples","curves"]:(output/directory).mkdir()
        run(args,output)
    except BaseException as error:
        status(output,"failed",error=repr(error))
        (output/"traceback.txt").write_text(traceback.format_exc())
        raise


if __name__=="__main__":main()
