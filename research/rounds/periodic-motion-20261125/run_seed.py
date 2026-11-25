"""Run both pre-registered carrier arms and the shared baseline for one seed."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

PROJECT = Path(__file__).resolve().parents[3]
MODEL = "experiments/autopilot_flight_dynamics/model.py"
LOW = "study-c636b4ae2d87/exp-7c2120e6fa14"
HISTORY = "study-f408239b1b1e/exp-ee674dbe15d9"
ARMS = {
    "actual-phase-carrier": ("3cb9e766707c6c89f2ece6e12c973b249db686cd", "起点实测拍频驱动的预测谐波",
        "用过去运动估计各轴两种正交系数与初始载波值，在各自时间流以起点拍频推进并施加可学的逐轴正衰减。"),
    "matched-slow-basis": ("38aa995b23b75752ddbaeec2785209babc048ab5", "固定慢时钟的匹配双分支",
        "保留系数网络、全部输入、两个正交维度、衰减和输出差分，只将载波的频率换成0.25 Hz。"),
}


def read(path):
    return json.loads(path.read_text())


def cli(sid, action, *args, form=None):
    command = ["research-autopilot", "experiment", action, sid, *args]
    print(" ".join(command), flush=True)
    with tempfile.TemporaryDirectory(prefix="periodic-plan-") as tmp:
        if form is not None:
            file = Path(tmp) / "form.json"
            file.write_text(json.dumps(form, ensure_ascii=False))
            command.append(str(file))
        completed = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"{action} failed with code {completed.returncode}: "
                           + (completed.stderr + completed.stdout)[-4800:])
    return json.loads(completed.stdout)


def plan_form(state, protocol, arm):
    commit, mechanism, change = ARMS[arm]
    registered = next(row for row in protocol["hypotheses"] if row["id"] == arm)
    prior = [json.loads(s) for s in (PROJECT / "research/memory/experiments.jsonl").read_text().splitlines()]
    failures = [r["experiment_id"] for r in prior if r.get("decision") in ("discard", "failed")
                and r.get("hypothesis", {}).get("mechanism") == mechanism]
    source = state.get("source") or {}
    checks = [m["id"] for m in source.get("mechanisms", [])]
    hypothesis = dict(
        hypothesis_id=arm, observation="上一轮强递推模型给定实录命令可预测角速度，但未检验拍频同步运动能否贡献净预测增益。",
        mechanism=mechanism, motivation="区分增加模型容量和引入与实际扑翼同步的时间基；只改变正交基频率。",
        modification=change, prediction=registered["prediction"], falsifier=registered["falsifier"],
        risk="起点频率不能完全刻画之后的拍频漂移。三轴机体速度还受到采样率和姿态坐标变换限制。",
        history_relation="保留上轮低阶命令潜态；三组共享决策时可用的历史运动谐波摘要。",
        cost=f"种子{state['seeds'][0]}，{protocol['device']}；完整训练至冻结平台或登记步数上限。",
        modules=[MODEL], evidence=[f"冻结候选源码 {commit} 和同初始化/两组谐波系数梯度检查。",
          f"论文交接 {source.get('reproduction_id', '见共同研究方案')}，角色是有结构时间基的来源；飞行分离尚待检验。"],
        target_metrics=list(protocol["metrics"]), reproduced_checks=checks, history_ids=failures)
    if state.get("from_cycle"):
        hypothesis["prior_refs"] = [LOW, HISTORY]
    if failures:
        hypothesis["new_evidence"] = "本实验属于同一冻结多种子研究，在事先指定的另一次初始化上验证原比较。"
    return dict(parent="baseline", selection_reason="两臂均从同一基线起步，并执行相同的特征、训练和开发评价协议。",
                alternatives=["普通单头与同输入的匹配容量双头控制均纳入实际比较。"], hypothesis=hypothesis)


def prepare_worktree(item, arm):
    tree = Path(item["worktree"])
    path = tree / MODEL
    wanted = subprocess.check_output(["git", "-C", str(tree), "show", ARMS[arm][0] + ":" + MODEL])
    if path.read_bytes() == wanted:
        head = subprocess.check_output(["git", "-C", str(tree), "rev-parse", "HEAD"], text=True).strip()
        if head != item["start_commit"]:
            return
    else:
        if subprocess.check_output(["git", "-C", str(tree), "status", "--porcelain"]):
            raise RuntimeError(f"Candidate worktree with unrelated changes: {tree}")
        path.write_bytes(wanted)
    for cmd in (["diff", "--check"], ["add", "--", MODEL],
                ["commit", "-m", ARMS[arm][1], "--", MODEL]):
        subprocess.run(["git", "-C", str(tree), *cmd], check=True)


def inspect_run(folder, item, protocol):
    seed, = item["result"]["seeds"]
    train = seed["training"]
    steps = train["steps"]
    assert steps >= 49152 and train["samples"] == steps * 384 and train["parameter_change_l2"] > 0
    if train["stop"] == "budget_cap":
        assert steps == protocol["steps"]
        outcome = "预算上限"
    else:
        assert train["stop"] == "sustained_plateau" and steps < protocol["steps"]
        rows = train["validation_history"]
        tail = rows[-16:]
        assert all(r["lr"] == .000125 for r in tail)
        assert max(r["dev_mse"] for r in tail) - min(r["dev_mse"] for r in tail) <= .002
        assert min(r["dev_mse"] for r in rows[:-16]) - min(r["dev_mse"] for r in tail) <= .001
        outcome = "验证平台"
    last = item["attempts"][-1]
    root = folder / last["directory"] / f"seed-{seed['seed']}" / "output"
    if not (root / "step49152.pt").is_file():
        raise RuntimeError(f"Missing common-budget checkpoint: {root}")
    return train, outcome


def decide_form(folder, state, protocol, item):
    result = item["result"]
    train, stop = inspect_run(folder, item, protocol)
    base = state["baseline"]["result"]
    score, ref = result["metrics"], base["metrics"]
    primary = protocol["primary"]
    primary_gain = result["delta_best"][primary]
    ready = result["training_completion"]["status"] == base["training_completion"]["status"] == "complete"
    eligible = not result["violations"] and primary_gain > 0 and primary_gain >= protocol["min_gain"]
    best = state["baseline"] if state["best"] == "baseline" else state["experiments"][state["best"]]
    comparison_capped = any(r["result"]["seeds"][0]["training"]["stop"] == "budget_cap"
                            for r in [state["baseline"], best])
    if stop == "预算上限" or comparison_capped or not ready:
        action = "investigate"
    else:
        action = "keep" if eligible else "discard"
    performance = "；".join(f"{k} {score[k]:.6f}，本种子基线 {ref[k]:.6f}" for k in score)
    return dict(experiment_id=item["id"], attempt=len(item["attempts"]), decision=action,
        outcome=f"终点主指标 {primary} {score[primary]:.6f}；相对基线收益 {result['delta_baseline'][primary]:+.6f}，"
                f"相对当前最佳收益 {primary_gain:+.6f}。约束违规 {result['violations']}。本种子工程判断 {action}。",
        components=performance, training=f"种子{state['seeds'][0]}，{train['steps']}次更新，"
        f"样本{train['samples']}；参数变化L2为{train['parameter_change_l2']:.5f}。停止依据：{stop}。"
        "共同49152步的权重已保存并单列评价。",
        confounds="同一次初始化与同样输入/训练协议，基线与两头之间有参数差异；两个插入头具有相同初始化和参数量。",
        supported="本开发集上可核对本种子的方向及预设速度护栏。跨种子与整架次配对见整轮汇总。",
        uncertain="实际慢运动和周期分量未被直接测量，须对高频原生观测、慢参考和泄漏进一步检验。",
        next="两臂完成后归档此种子；统一进行相同步数的配对和独立组分检验。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study")
    args = parser.parse_args()
    sid = args.study
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    folder = PROJECT / "research/experiments" / sid
    protocol = read(folder / "protocol.json")
    assert [r["id"] for r in protocol["hypotheses"]] == list(ARMS)
    assert protocol["primary"] == "w2" and len(protocol["seeds"]) == 1
    state = read(folder / "state.json")
    if state.get("closed_at"):
        print("Already closed", sid)
        return
    if state["baseline"]["status"] != "measured":
        cli(sid, "baseline")
    state = read(folder / "state.json")
    inspect_run(folder, state["baseline"], protocol)
    for arm in ARMS:
        state = read(folder / "state.json")
        existing = [v for v in state["experiments"].values() if v["hypothesis"]["hypothesis_id"] == arm]
        if existing and existing[0].get("decision"):
            continue
        item = existing[0] if existing else cli(sid, "plan", form=plan_form(state, protocol, arm))
        if item["status"] == "planned":
            prepare_worktree(item, arm)
            cli(sid, "run")
        state = read(folder / "state.json")
        item = state["experiments"][item["id"]]
        if item["status"] != "measured":
            raise RuntimeError(f"Unresolved {arm} state: {item['status']}")
        cli(sid, "decide", form=decide_form(folder, state, protocol, item))
    state = read(folder / "state.json")
    lines = ["共同基线和两个预注册模型已实测。以下为各自终点的2秒速度/角速度（m/s，rad/s）和更新次数。"]
    for item in [state["baseline"], *state["experiments"].values()]:
        result = item["result"]
        train, stop = inspect_run(folder, item, protocol)
        label = item.get("hypothesis", {}).get("hypothesis_id", "baseline")
        lines.append(f"{label} {result['metrics']['v2']:.6f}/{result['metrics']['w2']:.6f}，"
                     f"{train['steps']}步，{stop}。")
    lines.append("跨初始化的配对与分离证据在整轮汇总。")
    print("".join(lines), flush=True)
    cli(sid, "finish", "--conclusion", "".join(lines))


if __name__ == "__main__":
    main()
