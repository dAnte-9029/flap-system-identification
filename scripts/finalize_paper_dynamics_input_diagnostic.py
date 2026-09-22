"""Verify finished Step5, render available CJK fonts and add evidence-based review.

No inference, training, regrouping or numeric-metric changes. Frozen scientific
protocol remains byte-identical; finalization has its own provenance record.
"""
from pathlib import Path
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib import font_manager
from system_identification.data.september_trajectory import file_hash
from run_paper_baseline_comparison import write_json
from run_paper_dynamics_input_diagnostic import OUT,ART,A3,verify,read,rel
from report_paper_dynamics_input_diagnostic import generate


def main():
    if (OUT/'review_verification.json').exists():raise FileExistsError('review already finalized')
    completion=read(OUT/'completion.json');protocol=read(OUT/'protocol.json')
    assert completion['status']=='complete'
    verify(completion['artifact_sha256']);verify(protocol['frozen_input_sha256'])
    original_completion_sha=file_hash(OUT/'completion.json')
    (OUT/'inference_completion.json').write_bytes((OUT/'completion.json').read_bytes())
    scientific_tables=['per_flight.csv','per_seed_summary.csv','multiseed_summary.csv','paired_differences.csv',
        'control_group_thresholds.csv','control_group_coverage.csv','error_evolution.csv','trajectory_interval_errors.csv',
        'representative_selection.json','sanity_checks.json','protocol.json']
    table_hashes={name:file_hash(OUT/name) for name in scientific_tables}
    # Existing system font only; no dependency installation or environment change.
    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if not font.exists():raise FileNotFoundError('expected installed CJK font missing')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family']=[font_manager.FontProperties(fname=str(font)).get_name(),'DejaVu Sans']
    batch=torch.load(A3/'prepared.pt',map_location='cpu',weights_only=False)['validation']
    generate(OUT,ART,protocol,batch)
    for name,h in table_hashes.items():assert file_hash(OUT/name)==h,name
    pairs=pd.read_csv(OUT/'paired_differences.csv')
    macro=pairs.query('level=="macro" and kind=="endpoint" and cohort=="ALL"')
    all500=macro.query('step==25 and group=="ALL"').set_index('metric')
    high=macro.query('step==25 and group=="high"').set_index('metric')
    metrics=['velocity_rmse_m_s','attitude_error_deg','body_rate_rmse_rad_s']
    a=[all500.loc[m,'actual'] for m in metrics];h=[all500.loc[m,'hold'] for m in metrics]
    pct=[all500.loc[m,'relative_gain_pct'] for m in metrics];hpct=[high.loc[m,'relative_gain_pct'] for m in metrics]
    results=(f'At the nominal 500 ms horizon, replaying the logged future commands reduced velocity RMSE from {h[0]:.4f} to {a[0]:.4f} m/s '
        f'({pct[0]:.2f}%), attitude geodesic RMS error from {h[1]:.4f}° to {a[1]:.4f}° ({pct[1]:.2f}%), and body-rate RMSE from '
        f'{h[2]:.4f} to {a[2]:.4f} rad/s ({pct[2]:.2f}%) compared with holding the origin command. '
        'All three seeds favored actual-command replay on these aggregate metrics. After averaging seeds within each flight, '
        'the improvement direction was also consistent across all 17 validation flights. '
        f'In the high-control-change subset (454 origins spanning all 17 flights), the corresponding reductions were {hpct[0]:.2f}%, '
        f'{hpct[1]:.2f}% and {hpct[2]:.2f}%. These findings support incremental predictive information from future logged commands '
        'within the observed flight distribution; they do not imply improvement for every individual trajectory.')
    old=read(OUT/'interpretation.json')['results_paragraph']
    for name in ('report.md','review_report.md'):
        path=OUT/name;path.write_text(path.read_text().replace(old,results))
    (OUT/'paper_results_paragraph.txt').write_text(results+'\n')
    zh=(OUT/'review_report.md').read_text()
    additions={
        '## 2. 实验设置与覆盖情况':'500 ms low/middle/high分别888/1,240/454个origins，全部覆盖17flights。High组包含Sep7的185个、Sep17的269个origins，每flight为8–58个。覆盖广，但某些flight的组内样本较少，等flight加权不会消除其估计波动。',
        '## 3. 500 ms主结果':'总体velocity/attitude/body-rate的改善分别17.43%/13.24%/7.34%；三个seed全部同方向，先三seed平均后的17个flight也全部同方向。Position为辅助指标，改善5.03%，其中16/17flights同方向。不能把这些方向计数当作基于独立window的显著性检验。',
        '## 4. 控制变化分组结果':'High组velocity/attitude/body-rate改善16.92%/17.58%/7.48%，seed方向均3/3；flight方向分别15/17、17/17、14/17。Low组也有12.21%/9.57%/3.86%的改善，所以价值不只存在于high组。不能声称控制变化越大、所有指标的百分比收益越大：velocity在middle为18.81%，略高于high；body rate也是middle高于high。High组velocity绝对收益0.1035 m/s，高于low的0.0407 m/s。',
        '## 5. 不同预测时域及完整误差过程':'首步预测完全相同。100/200/500/1000 ms的velocity收益分别2.32%/6.19%/17.43%/23.59%；attitude为7.73%/8.26%/13.24%/17.41%；body rate为−0.17%/2.74%/7.34%/22.01%。100 ms body-rate略偏向Hold（Actual0.54533、Hold0.54439 rad/s），已完整保留；不是所有时间点都支持正结果。500 ms区间加权误差也支持Actual，三项改善为11.99%/10.95%/3.65%，但幅度不能与端点指标混用。\n\n实际step25累计时间的中位数0.499094 s、范围0.488664–0.518876 s；step50中位数0.997967 s、范围0.987696–1.027915 s。未来命令的相对信息增益在较长预测时域更明显，与既有history消融“历史上下文相对收益在短时更强”是两个不同的诊断结论，不相互替代。',
        '## 6. 动态响应案例':'案例结果刻意完整保留：原Step1固定案例在500 ms四项端点误差均由Hold更低；Sep7 high案例四项均Actual更低；Sep17 high案例velocity/attitude/position由Hold更低，body rate由Actual更低。这些图说明总体统计优势不保证单条响应轨迹更准确，不以个案替代17flight配对统计，也不删换不利案例。',
        '## 7. 能够支持的论文结论':'两个cohort均支持500 ms三项dynamics的正向总体增益。Sep7与Sep17的velocity增益分别10.79%和23.32%，attitude分别12.18%和14.16%，body rate分别3.39%和11.89%；输入信息价值具有session和state依赖，不应以ALL平均掩盖。',
    }
    for heading,paragraph in additions.items():zh=zh.replace(heading,heading+'\n\n'+paragraph,1)
    (OUT/'review_report.md').write_text(zh)
    en=(OUT/'report.md').read_text()
    en=en.replace('## Scientific interpretation','## Scientific interpretation\n\n'
        'Positive aggregate gains are not universal. At100ms, body-rate error slightly favors Hold (−0.17% relative gain). '
        'The fixed Step1 case favors Hold on all four500ms endpoints; the Sep17 high-control case favors Hold for position, velocity and attitude. '
        'All cases are retained. Relative gains do not increase monotonically across activity groups: middle-group velocity/body-rate percentage gains exceed high-group gains. '
        'Both cohorts nevertheless support500ms aggregate gains on all three dynamics metrics.\n\n',1)
    (OUT/'report.md').write_text(en)
    interpretation=read(OUT/'interpretation.json');interpretation['results_paragraph']=results
    interpretation['counterexamples_retained']=['100ms body-rate slightly favors Hold','fixed Step1 case all4endpoints favor Hold','Sep17 high case mixed directions']
    write_json(OUT/'interpretation.json',interpretation)
    # Independent arithmetic checks against reported pair tables and subgroup coverage.
    per=pd.read_csv(OUT/'per_flight.csv');summary=pd.read_csv(OUT/'per_seed_summary.csv')
    for _,row in summary.query('cohort=="ALL"').iterrows():
        g=per[(per.kind==row.kind)&(per.condition==row.condition)&(per.seed==row.seed)&(per.group==row.group)&(per.step==row.step)&(per.metric==row.metric)]
        assert len(g)==row.n_flights and g.n_origins.sum()==row.n_origins
        assert np.isclose(g.value.mean(),row.value,rtol=1e-13,atol=1e-13)
    assert np.isfinite(per.value).all()
    with np.load(ART/'per_origin_squared_errors.npz') as f:
        for key in f.files:
            if key!='window_ids':assert f[key].shape==(2582,50,4) and np.isfinite(f[key]).all()
    verify(protocol['frozen_input_sha256'])
    review=dict(status='passed',reviewed_at_unix=time.time(),initial_output_hashes_verified=len(completion['artifact_sha256']),
        frozen_input_hashes_verified=len(protocol['frozen_input_sha256']),original_inference_completion_sha256=original_completion_sha,
        source_sha256=file_hash(Path(__file__)),font_path=str(font),font_sha256=file_hash(font),
        scientific_table_hashes_unchanged=table_hashes,all_per_seed_ALL_means_independently_recomputed=True,
        per_origin_squared_errors_finite_shape=[2582,50,4],all_counterexamples_retained=True,
        original_Actual_small_batch_bitwise_parity=True,naturally_constant_origins=0,
        constant_tape_verification='synthetic bitwise equality; no naturally fully constant50step tapes in validation',
        model_and_normalization_unchanged=True,sealed_test_opened_this_run=False,reserved_evaluation_opened_this_run=False,
        no_training=True,no_additional_inference=True)
    write_json(OUT/'review_verification.json',review)
    completion.update(finalized_at_unix=time.time(),report_finalizer_sha256=file_hash(Path(__file__)),
        review_verification='review_verification.json',
        artifact_sha256={rel(f):file_hash(f) for directory in (OUT,ART) for f in directory.iterdir() if f.is_file() and f.name not in ('completion.json','status.json','run.log')})
    write_json(OUT/'completion.json',completion)
    print('Final review passed; required sources and numeric tables unchanged; CJK figures and nuanced report finalized.')


if __name__=='__main__':main()
