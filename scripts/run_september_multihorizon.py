#!/usr/bin/env python3
"""Matched one-second, two-second and joint free-run objectives."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

import numpy as np
import pandas as pd
import torch

from run_september_step2 import prepare, evaluate, summarize_curves, plot, status, write_json
from system_identification.data.september_trajectory import file_hash
from system_identification.training.trajectory_main_v1 import (
    MainV1Config, assemble_history_trajectory_windows, fit_main_v1_stats, fit_history_trajectory_model,
)


def prepare_matched(registry):
    manifest,entry,samples,windows,coverage=prepare(registry,train_horizon=2)
    old=ROOT/"artifacts/september_step2_20260910/validation_selected_windows.parquet"
    reference=pd.read_parquet(old)
    keys=["window_id","log_id","segment_id","start_sample_in_log","end_sample_in_log",
          "start_timestamp_us","end_timestamp_us"]
    if not windows["validation"][keys].equals(reference[keys]):
        raise ValueError("validation origins differ from Step 2")
    return manifest,entry,samples,windows,coverage


def gains(curves):
    metrics=[c for c in curves if "rmse" in c]
    selected=curves.loc[curves.global_horizon_s.isin([1,2,3,5])]
    reference=selected.loc[selected.model=="matched_1s"].set_index(["log_id","global_horizon_s"])
    rows=[]
    for name in ["full_2s","joint_1s_2s"]:
        candidate=selected.loc[selected.model==name].set_index(["log_id","global_horizon_s"])
        for index,row in candidate.iterrows():
            result={"model":name,"reference":"matched_1s","log_id":index[0],"horizon_s":index[1]}
            for metric in metrics:
                base=float(reference.loc[index,metric]); value=float(row[metric])
                result[metric+"_gain_percent"]=100*(base-value)/base if base>0 else np.nan
            rows.append(result)
    return pd.DataFrame(rows)


def run(args,output):
    if not torch.cuda.is_available() or torch.cuda.device_count()<2:
        raise RuntimeError("GPU 1 unavailable; no CPU fallback")
    torch.set_num_threads(4);torch.manual_seed(17)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    status(output,"loading_and_verifying_data")
    manifest,entry,samples,windows,coverage=prepare_matched(args.registry)
    coverage.to_csv(output/"coverage.csv",index=False)
    for part,w in windows.items():w.to_parquet(output/f"{part}_selected_windows.parquet",index=False)
    common=dict(use_history=True,use_controls=True,hidden_size=64,epochs=40,batch_size=256,
                learning_rate=3e-4,weight_decay=1e-5,gradient_clip_norm=5,seed=17)
    configs=[MainV1Config(model_name="matched_1s",objective_steps=50,**common),
        MainV1Config(model_name="full_2s",objective_steps=100,**common),
        MainV1Config(model_name="joint_1s_2s",objective_steps=100,
                     prefix_loss_weights=((50,.5),(100,.5)),**common)]
    paths=[Path(__file__),ROOT/"scripts/run_september_step2.py",
        ROOT/"src/system_identification/training/trajectory_main_v1.py",
        ROOT/"src/system_identification/models/trajectory_main_v1.py",
        ROOT/"src/system_identification/evaluation/trajectory_rollout_diagnostics.py",
        ROOT/"src/system_identification/evaluation/trajectory.py",
        ROOT/"src/system_identification/training/trajectory_baselines.py",
        ROOT/"src/system_identification/models/trajectory.py"]
    run_manifest={"experiment":"september_multihorizon","dataset":entry,"dataset_id":manifest["dataset_id"],
        "split_contract":manifest["split_contract"],"sampling":manifest["sampling"],
        "future_known":manifest["roles"]["known_future_control_t0_to_tT_exclusive"],
        "future_forbidden":manifest["roles"]["future_forbidden_as_input"],
        "phase":"unchanged relative-to-origin representation; no absolute phase input",
        "configs":[asdict(c) for c in configs],"device":"cuda:1","python_executable":sys.executable,
        "train_windows":len(windows["train"]),"validation_windows":len(windows["validation"]),
        "common_training_origins":True,"validation_origins_equal_step2":True,
        "common_normalization":"fit once on common two-second training batch and train transitions",
        "validation_used_for_fitting_or_early_stopping":False,"sealed_test_opened":False,
        "source_hashes":{str(p.relative_to(ROOT)):file_hash(p) for p in paths},
        "protocol":"docs/contracts/2026-09-11_september_multihorizon.md",
        "git_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "git_status":subprocess.check_output(["git","status","--short"],cwd=ROOT,text=True)}
    write_json(output/"manifest.json",run_manifest)
    train=assemble_history_trajectory_windows(samples["train"],windows["train"],history_steps=26)
    if not train.history_mask.all():raise ValueError("unexpected padded train history")
    stats=fit_main_v1_stats(samples["train"],train)
    np.savez(output/"normalization.npz",**vars(stats))
    models={};budgets=[]
    for config in configs:
        status(output,"training",model=config.model_name)
        torch.cuda.synchronize(1);torch.cuda.reset_peak_memory_stats(1)
        start=time.monotonic()
        model,history=fit_history_trajectory_model(train,stats,config,device="cuda:1")
        torch.cuda.synchronize(1)
        budgets.append({"model":config.model_name,"training_wall_s":time.monotonic()-start,
            "optimizer_updates":config.epochs*math.ceil(len(windows["train"])/config.batch_size),
            "nominal_rollout_transitions":config.epochs*len(windows["train"])*config.objective_steps,
            "peak_allocated_gpu_bytes":torch.cuda.max_memory_allocated(1)})
        pd.DataFrame(budgets).to_csv(output/"training_budget.csv",index=False)
        history.to_csv(output/f"training_history_{config.model_name}.csv",index=False)
        torch.save({"state_dict":model.state_dict(),"config":asdict(config),"history_steps":26},
                   output/"models"/f"{config.model_name}.pt")
        models[config.model_name]=model
    del train
    status(output,"assembling_validation")
    batch=assemble_history_trajectory_windows(samples["validation"],windows["validation"],history_steps=26)
    if not batch.history_mask.all():raise ValueError("unexpected padded validation history")
    curves=[];endpoints=[]
    for name,model in models.items():
        status(output,"evaluating",model=name)
        curve,endpoint=evaluate(name,model,samples["validation"],windows["validation"],batch,output,"continuous",0)
        curves.append(curve);endpoints.append(endpoint)
    curves=pd.concat(curves,ignore_index=True);endpoints=pd.concat(endpoints,ignore_index=True)
    curves.to_csv(output/"per_log_error_curves.csv",index=False)
    macro=summarize_curves(curves)
    macro.to_csv(output/"equal_log_error_curves.csv",index=False)
    macro.loc[macro.global_horizon_s.isin([1,2,3,5])].to_csv(output/"endpoint_metrics.csv",index=False)
    endpoints.to_parquet(output/"window_endpoint_metrics.parquet",index=False)
    perlog=gains(curves);perlog.to_csv(output/"per_log_matched_gains.csv",index=False)
    win_counts=perlog.groupby(["model","horizon_s"])[[c for c in perlog if c.endswith("gain_percent")]].agg(lambda x:int((x>0).sum()))
    win_counts.to_csv(output/"improved_log_counts.csv")
    macro_as_log=macro.assign(log_id="equal_log_macro")
    gains(macro_as_log).to_csv(output/"matched_macro_gains.csv",index=False)
    previous=pd.read_csv(ROOT/"artifacts/september_step2_20260910/endpoint_metrics.csv")
    previous.loc[(previous.model=="history_controlled")&(previous['mode']=="continuous")].to_csv(
        output/"step2_context_only_different_training_set.csv",index=False)
    plot(macro,output)
    write_json(output/"summary.json",{"status":"completed","experiment":run_manifest["experiment"],
        "train_windows":len(windows["train"]),"validation_windows":len(windows["validation"]),
        "models":list(models),"sealed_test_opened":False,
        "interpretation":"single-seed matched training-objective diagnostic; no automatic promotion"})
    status(output,"completed")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry",type=Path,default=ROOT/"configs/data/trajectory_dataset_registry.yaml")
    parser.add_argument("--output-root",type=Path,default=ROOT/"artifacts/september_multihorizon_20260911")
    parser.add_argument("--preflight",action="store_true")
    args=parser.parse_args()
    if args.preflight:
        _,_,_,windows,coverage=prepare_matched(args.registry)
        torch.ones(1,device="cuda:1")
        print(coverage.to_string(index=False));print({p:len(w) for p,w in windows.items()})
        return
    output=args.output_root.resolve();output.mkdir(parents=True,exist_ok=False)
    try:
        for name in ["models","errors","examples","curves"]:(output/name).mkdir()
        run(args,output)
    except BaseException as error:
        status(output,"failed",error=repr(error))
        (output/"traceback.txt").write_text(traceback.format_exc())
        raise


if __name__=="__main__":main()
