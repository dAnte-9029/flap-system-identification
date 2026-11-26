"""Summarize the frozen evaluation, preserving empty phase cells and flight pairing."""
from pathlib import Path
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parents[2]/'research/rounds/wingbeat-scale-exchange/extended'
NAMES={'phase_main':'既有相位主模型','ode':'裸 ODE','fixed_self':'固定频率＋自身驱动','clock_self':'起点拍频＋自身驱动','fixed_cross':'固定频率＋快到慢','full':'首组完整','clock_deleted':'首组删除快到慢','reference':'强基线 ODE＋波形头','deleted':'删除乘积项','self':'等容量自身调制','cross':'最终跨尺度调制'}


def main():
    ep=pd.read_csv(OUT/'endpoints.csv')
    ha=pd.read_csv(OUT/'harmonics.csv')
    ef=pd.read_csv(OUT/'endpoint_flights.csv').melt(id_vars=['arm','flight','horizon','metric'],var_name='axis')
    ef['period']=ef.horizon.map({.5:'0.5s',1.:'1s',2.:'2s'})
    ef['order']=0
    hf=pd.read_csv(OUT/'harmonic_flights.csv').rename(columns={'mean':'value'})
    by_flight=pd.concat([ef,hf],ignore_index=True)
    rows=[]
    for key,sub in by_flight.groupby(['period','axis','order','metric']):
        table=sub.pivot(index='flight',columns='arm',values='value')
        assert table.notna().all().all()
        draws=np.random.default_rng(17).integers(0,len(table),(10000,len(table)))
        for target,ref in [('cross','self'),('cross','deleted'),('cross','reference'),('cross','phase_main'),('full','fixed_cross')]:
            gain=(table[ref]-table[target]).to_numpy()
            if key[-1]=='PCC':
                gain=-gain
            lo,hi=np.quantile(gain[draws].mean(1),[.025,.975])
            rows.append([*key,target,ref,len(table),gain.mean(),lo,hi,(gain>0).sum()])
    pd.DataFrame(rows,columns=['period','axis','order','metric','model','reference','flights','improvement','lo95','hi95','wins']).to_csv(OUT/'paired.csv',index=False)
    lines=['# 冻结模型新增指标','', '种子17。端点在39个开发架次、44,524个窗口上重算，所有臂均回放通过原有六项成绩核对。速度为起点FRD增量（m/s），角速度为机体系分量增量（rad/s）。误差和P95均按架次等权。', '']
    def table(head,body):
        lines.extend(['| '+' | '.join(head)+' |','|'+'---|'*len(head)])
        lines.extend('| '+' | '.join(r)+' |' for r in body)
        lines.append('')
    for h in [.5,1.,2.]:
        lines.extend([f'## {h:g} 秒原生端点','','每格为 MAE / 本架次绝对误差第95百分位的架次均值。',''])
        vals=ep[ep.horizon==h].set_index(['arm','metric'])
        table(['模型','vx','vy','vz','滚转 p','俯仰 q','偏航 r'],
              [[label,*[f"{vals.loc[(arm,'MAE'),c]:.4f} / {vals.loc[(arm,'P95'),c]:.4f}" for c in ('vx','vy','vz','p','q','r')]] for arm,label in NAMES.items()])
    lines.extend(['## 早期完整翼拍','','沿用冻结的完整拍窗，以拍内原生点的加权中心时刻划入 ≤0.5 秒组。角速度包含193个拍窗、35个架次。Pearson 相关按拍内原生点加权，先在架次内平均。各阶幅值误差使用全部合格拍。相角只统计实测幅值 ≥0.10 rad/s 的可辨拍；角度表示该阶谐波相角。该阈值为分析选择。相位基准和未来真值只在评分时使用。',''])
    vals=ha[ha.period=='early'].set_index(['arm','axis','metric','order'])
    for axis in ['p','q','r']:
        coverage=[]
        for k in [1,2,3]:
            row=vals.loc[('cross',axis,'phase_MAE_deg',k)] if ('cross',axis,'phase_MAE_deg',k) in vals.index else None
            coverage.append(f'{int(row.beats)}拍/{int(row.flights)}架' if row is not None else '0拍/0架')
        lines.extend([f'### {axis} 轴','','一、二、三阶相位支持分别为 '+'、'.join(coverage)+'。每阶列为幅值MAE（rad/s）/ 相位MAE（°），— 表示无合格观测。',''])
        body=[]
        for arm,label in NAMES.items():
            r=[label,f"{vals.loc[(arm,axis,'PCC',0),'value']:.3f}"]
            for k in (1,2,3):
                phi=(arm,axis,'phase_MAE_deg',k)
                ph=f"{vals.loc[phi,'value']:.1f}°" if phi in vals.index else '—'
                r.append(f"{vals.loc[(arm,axis,'amplitude_MAE',k),'value']:.4f} / {ph}")
            body.append(r)
        table(['模型','PCC','一阶','二阶','三阶'],body)
    lines.extend(['速度逐轴谐波、中期与晚期角速度、各项覆盖见[分轴分阶全表](harmonics.csv)。缺失的相位组合表示0个合格拍。原生点上的翼拍验证使用原有筛选与加权拟合，不能从八个稀疏端点插值得到。各新增指标是事后开发集诊断。','',
                  '[逐架次配对差与描述性95%区间](paired.csv) · [全时域端点表](endpoints.csv) · [评分口径](protocol.md)'])
    (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
