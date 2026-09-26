"""Record verified descriptive conclusions without changing analysis registration."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import run_response_timescale_diagnostic as r
m=r.m;OUT=r.OUT

def main():
    p=m.read_json(OUT/'protocol.json');m.verify_pins(p['frozen_input_sha256'])
    c=m.read_json(OUT/'completion.json');m.verify_pins(c['artifact_sha256'])
    f=pd.read_csv(OUT/'component_per_flight.csv')
    keys=['seed','tau_s','interval','group','flight_id','cohort','axis']
    wide=f.pivot(index=keys,columns='component',values='error_mse')
    cross=f[f.component=='raw'].set_index(keys).cross_term
    np.testing.assert_allclose(wide.raw,wide.slow+wide.fast+cross,rtol=1e-11,atol=1e-11)
    raw=f.query('component=="raw"').pivot(index=[k for k in keys if k!='tau_s'],columns='tau_s',values='rmse')
    np.testing.assert_array_equal(raw[.05],raw[.1]);np.testing.assert_array_equal(raw[.1],raw[.2])
    coverage=pd.read_csv(OUT/'excitation_per_flight.csv')
    assert (coverage.positive+coverage.negative+coverage.zero==coverage.n_origins).all()
    for key,g in coverage.groupby(['partition','channel','flight_id']):
        x=g.set_index('subset').n_origins;assert x.isolated<=x.predominant<=x.high
    interpretation='''
## 结果解读与下一步判断
在预先指定的100ms分析尺度、高控制变化组、0–500ms区间：p原始误差RMS为0.5315 rad/s，慢分量0.1819、快分量0.4429；p的预测/实飞变化幅值比慢分量0.962、快分量0.654。慢分量并非已经准确，其误差仍约为真实慢变化幅值的32.4%；快残差相对误差约84.9%。这支持滚转误差存在突出的较快波形匹配问题，而不是所有持续滚转变化都按同一比例偏弱。不能把快残差RMS平方直接除以原始MSE并称为独立误差贡献率，因为交叉项非零。

r原始误差RMS为0.1622 rad/s，慢分量0.1071、快分量0.0971；幅值比分别0.862、0.689，慢分量相对误差52.7%。所以偏航问题不能归结为较快残差：较慢运动变化也存在明显偏差。q作为对照，慢/快幅值比分别0.945/0.884，相对误差25.5%/45.6%。

50/100/200ms尺度敏感性：p慢分量幅值比0.939/0.962/0.975、快分量0.595/0.654/0.701；r慢分量0.857/0.862/0.862、快分量0.620/0.689/0.748。结论不是只在某个滤波尺度出现。两个cohort中p慢幅值比较接近1，而r慢幅值偏低：100ms尺度p在Sep7/Sep17约0.973/0.950，r约0.898/0.831。500–1000ms区间p/r慢幅值比约0.953/0.854，没有得到“问题只在最初瞬态”的支持。

validation严格isolated窗口：drive23个/13flights，共模18个/10flights，差模14个/6flights，rudder4个/2flights。rudder仅正向4个、负向0个，且全部来自Sep7；Sep17的严格共模仅2个、差模仅1个、rudder为0。这不能支持跨cohort、正负平衡的简单隔离响应检验。

在训练集中对应严格窗口分别199/96/98/59个，覆盖38/32/29/26flights。但train窗口重叠，不能把这些数目当成独立激励次数；这也不证明训练完全没有有效信息。放宽到predominant，validation可得到305/164/176/138个，仍只是“其他通道没有达到high门槛”，不是其他通道保持不变，更不是外生随机激励。

当前证据支持：优先区分p的较快波形残差来源，并检查r的较慢响应；不能将所有角速度误差都解释成执行器持续舵效不足。当前数据的角速度来自对齐状态网格，缺少逐周期完整高频输入，快残差可能包含真实扑翼/气动运动、状态估计或测量成分；本轮不能确定各自占比，尤其不能直接删掉快分量当作噪声。

建议先在下一项获授权工作中核对已准入开发日志的角速度源时间戳、记录频率和滤波链路，再明确是否需要同步记录/专门激励。若目标是验证独立操纵因果响应，需补充有批准幅度、明确事件、正负平衡、可重复工况及执行器记录的试验，特别是rudder和差模；仅改网络结构不弥补这些证据缺口。本轮未执行该采集或结构实验，没有调模型、滤波训练输入、控制器或执行器tau。
'''
    marker='\n## 结果解读与下一步判断\n'
    for name in ['report.md','review_report.md']:
        path=OUT/name;path.write_text(path.read_text().split(marker)[0]+interpretation)
    m.write_json(OUT/'tests.json',dict(status='passed',pytest='5 passed in 1.19s',checks=['native irregular-dt analytic step','constant signal','causal prefix','error cross-term identity','isolated-vs-coactive selection','activity independent of measured response'],
        per_flight_decomposition_cells=len(wide),raw_errors_invariant_across_filter_scales=True,coverage_counts_and_subset_nesting=True))
    c['artifact_sha256']={m.rel(path):m.file_hash(path) for path in OUT.iterdir() if path.is_file() and path.name!='completion.json'}
    c['finalizer_sha256']=m.file_hash(Path(__file__))
    m.write_json(OUT/'completion.json',c)
    print('VERIFIED',len(wide),'per-flight decompositions')

if __name__=='__main__':main()
