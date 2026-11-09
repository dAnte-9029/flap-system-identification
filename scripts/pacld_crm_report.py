"""Render fixed CRM diagnostics and Chinese report, without opening any dataset."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pacld_crm_v1 import OUT, BASE, weights, save

def supplements():
    df=pd.read_csv(OUT/'windows.csv');rng=np.random.default_rng(1702);m=df.moment_normal.to_numpy();M=weights(df)@m
    # Randomly rotated balanced date allocations; within date balanced flight allocations.
    # All slots marginally uniform in date, flight and window; no outcome used for allocation.
    dates=sorted(df.date.unique());flight_rows=[[g.index.to_numpy() for _,g in df[df.date==d].groupby('flight')] for d in dates]
    r=df.R_normal.to_numpy()/json.loads((OUT/'moments.json').read_text())['scales']['s_u'][0]
    sp=json.loads((OUT/'moments.json').read_text())['scales']['s_p']
    wr=weights(df)@r
    save('bias_escape.json',dict(weighted_normalized_residual_mean=float(wr),constant_increment_shift_to_cancel_global_moment=float(M*sp/wr),derivative_M_wrt_constant_increment_bias=float(-wr/sp)))
    records=[];composition=[]
    for B in [32,128,256,512]:
        est=[];nfl=[]
        for _ in range(10000):
            dc=np.bincount(np.resize(rng.permutation(len(dates)),B),minlength=len(dates));ids=[]
            for d,count in enumerate(dc):
                fs=flight_rows[d];chosen=np.resize(rng.permutation(len(fs)),count)
                ids.extend(int(rng.choice(fs[f])) for f in chosen)
            est.append(m[ids].mean());nfl.append(df.flight.iloc[ids].nunique())
        est=np.array(est);sq=est**2
        records.append(dict(scheme='date_flight_stratified',B=B,repeats=len(est),M_global=M,design_expectation=M,mean=est.mean(),bias_to_global=est.mean()-M,sd=est.std(),sign_flip=np.mean(est*M<0),M2_mean=sq.mean(),M2_q05=np.quantile(sq,.05),M2_q50=np.median(sq),M2_q95=np.quantile(sq,.95),variance_identity_error=abs(sq.mean()-est.mean()**2-est.var()),mean_unique_flights=np.mean(nfl)))
        median=np.median(nfl)
        for label,mask in [('fewer_flights',np.array(nfl)<=median),('more_flights',np.array(nfl)>median)]:
            if mask.any():composition.append(dict(B=B,composition=label,n=int(mask.sum()),mean_unique_flights=float(np.mean(np.array(nfl)[mask])),mean=float(est[mask].mean()),sd=float(est[mask].std()),sign_flip=float(np.mean(est[mask]*M<0))))
    pd.DataFrame(records).to_csv(OUT/'batch_stratified.csv',index=False);pd.DataFrame(composition).to_csv(OUT/'batch_composition.csv',index=False)

def render():
    moments=json.loads((OUT/'moments.json').read_text());tot=moments['moments'];audit=json.loads((OUT/'audit.json').read_text());res=json.loads((OUT/'residual_summary.json').read_text())
    windows=pd.read_csv(OUT/'windows.csv')
    flight_stats=windows.groupby(['date','flight']).moment_normal.agg(['mean',lambda x:np.var(x.to_numpy())]).reset_index()
    flight_stats.columns=['date','flight','mean','within_variance']
    date_stats=flight_stats.groupby('date').agg(mean=('mean','mean'),between_flight=('mean',lambda x:np.var(x.to_numpy())),within=('within_variance','mean'))
    variance_parts=dict(between_date=float(np.var(date_stats['mean'])),between_flight=float(date_stats.between_flight.mean()),within_flight=float(date_stats.within.mean()))
    save('sampling_variance_components.json',variance_parts)
    days=pd.read_csv(OUT/'date_moments.csv');flights=pd.read_csv(OUT/'flight_moments.csv');batch=pd.concat([pd.read_csv(OUT/'batch_statistics.csv'),pd.read_csv(OUT/'batch_stratified.csv')])
    identities=[]
    for row in batch[batch.scheme!='date_flight_stratified'].itertuples():
        prob=np.full(len(windows),1/len(windows)) if row.scheme=='window_uniform' else weights(windows)
        m=windows.moment_normal.to_numpy();mu=prob@m;var=prob@((m-mu)**2)/row.B
        identities.append(dict(scheme=row.scheme,B=row.B,target_M=float(mu),theoretical_batch_variance=float(var),theoretical_mean_squared_moment=float(mu**2+var),empirical_mean_squared_moment=row.M2_mean,monte_carlo_difference=float(row.M2_mean-mu**2-var)))
    pd.DataFrame(identities).to_csv(OUT/'batch_variance_identity.csv',index=False)
    grad=pd.read_csv(OUT/'gradient_statistics.csv');stable=pd.read_csv(OUT/'gradient_stability.csv');checks=json.loads((OUT/'gradient_checks.json').read_text());condition=pd.read_csv(OUT/'condition_moments.csv')
    fig,ax=plt.subplots(figsize=(11,5));xx=np.arange(len(days));ax.bar(xx,days.moment_normal,color=np.where(days.moment_normal>=0,'steelblue','tomato'));ax.axhline(tot['moment_normal']['M'],color='black',label='date/flight equal mean');ax.axhspan(*tot['moment_normal']['ci95'],color='gray',alpha=.2,label='date bootstrap 95% descriptive interval');ax.axhline(0,color='black',lw=.5);ax.set_xticks(xx,days.date.str[5:],rotation=55);ax.set_ylabel('Normalized signed error moment');ax.legend();fig.tight_layout();fig.savefig(OUT/'date_moments.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5));names=['normal','future50','future100'];labels=['0-100 ms','250-300 ms','250-350 ms'];vals=np.array([tot['moment_'+n]['M'] for n in names]);ci=np.array([tot['moment_'+n]['ci95'] for n in names]);axes[0].errorbar(np.arange(3),vals,yerr=np.c_[vals-ci[:,0],ci[:,1]-vals].T,fmt='o',capsize=5);axes[0].set_xticks(range(3),labels);axes[0].axhline(0,color='gray');axes[0].set_ylabel('Moment (own residual RMS)')
    for n,label in zip(names,labels):axes[1].plot(xx,days['moment_'+n],'.-',label=label)
    axes[1].set_xticks(xx,days.date.str[5:],rotation=60);axes[1].axhline(0,color='gray');axes[1].legend();fig.tight_layout();fig.savefig(OUT/'negative_controls.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for scheme,g in batch.groupby('scheme'):
        axes[0,0].plot(g.B,g.sd,'o-',label=scheme);axes[0,1].plot(g.B,g.sign_flip,'o-',label=scheme)
    axes[0,0].axhline(abs(vals[0]),color='gray',ls='--',label='global |M|');axes[0,0].set_ylabel('SD of batch moment');axes[0,0].legend(fontsize=8);axes[0,1].set_ylabel('Sign flip probability')
    mods=['coupling','equilibrium','relaxation','slow','decoder','belief']
    for kind in ['FULL','M','M_stop']:
        gg=grad[grad.objective==kind].groupby('module').rms.median();axes[1,0].plot(mods,[gg[m] if gg[m]>0 else np.nan for m in mods],'o-',label=kind)
    axes[1,0].text(.97,.05,'Stopped Belief gradient = 0',transform=axes[1,0].transAxes,ha='right',fontsize=8)
    axes[1,0].set_yscale('log');axes[1,0].set_ylabel('Median parameter gradient RMS');axes[1,0].legend()
    for kind in ['M','M2','M2_stop']:
        gg=stable[stable.objective==kind].groupby('module').cosine.median();axes[1,1].plot(mods,[gg[m] for m in mods],'o-',label=kind)
    axes[1,1].axhline(0,color='gray');axes[1,1].set_ylabel('Median between-batch gradient cosine');axes[1,1].legend()
    for a in axes[0]:a.set_xlabel('Effective batch size')
    for a in axes[1]:a.tick_params(axis='x',rotation=35)
    fig.tight_layout();fig.savefig(OUT/'batch_gradient_stability.png',dpi=170);plt.close(fig)
    def table(frame):
        def fmt(x):
            return f'{x:.6g}' if isinstance(x,(float,np.floating)) else str(x)
        return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |', '| '+' | '.join(['---']*len(frame.columns))+' |']+['| '+' | '.join(fmt(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None)])
    mt=pd.DataFrame([dict(metric=k.removeprefix('moment_'),M=v['M'],low=v['ci95'][0],high=v['ci95'][1],positive_dates=v['positive_dates']) for k,v in tot.items()])
    gs=grad.groupby(['objective','module'])[['l2','rms','cosine_FULL','max_abs']].median().reset_index()
    st=stable.groupby(['objective','module']).cosine.agg(['min','median','max']).reset_index()
    energy=pd.read_csv(OUT/'residual_distributions.csv');normal=energy[energy.control=='normal']
    corr=pd.read_csv(OUT/'residual_feature_correlations.csv');corr=corr[corr.control=='normal'].copy();corr['abs_corr']=corr.correlation.abs()
    selected_corr=corr.sort_values('abs_corr',ascending=False).head(10)
    groups=condition[condition.metric=='moment_normal']
    influence=json.loads((OUT/'influence.json').read_text())
    batch_checks=pd.DataFrame(checks['batches'])
    max_parity=max(x['factual_parity_max'] for x in checks['batches']);max_chain=max(x['autograd_M2_chain_max'] for x in checks['batches'])
    bias=json.loads((OUT/'bias_escape.json').read_text())
    verdict=dict(verdict='CONDITIONAL GO',A='VERIFIED: positive observational error moment on 16/17 dates',B='SUPPORTED: date/flight persistence with tail sensitivity; no independent-window inference',C='Reachability VERIFIED; stable useful full-train optimization direction UNRESOLVED; coverage batches do not identify it',D='Strong future-control negative moments preclude causal/control-gain supervision',route_A='significant confounding risk; no formal control-gain training recommendation',minimum_batch='B=512 is the strongest tested conservative candidate; not a proved universal minimum; B=32 unreliable',no_training=True)
    save('conclusion.json',verdict)
    text=f'''# PACLD-CRM-v1 控制残差矩监督可行性检验

结论：**CONDITIONAL GO，仅可作为观察性预测约束的受限等预算试验候选；不支持把它当作准确控制增益监督。路线A存在显著混杂风险，不建议直接进入正式控制增益训练。** 本轮没有训练、优化器、参数更新、MPC或validation/test数据访问。

## 数据、身份与估计口径

基线 `{audit['baseline']}`；冻结checkpoint SHA256 `{audit['checkpoint_sha256']}`。使用A2-v2全部 {audit['windows']} 个train窗口、{audit['flights']} 航次、{audit['dates']} 日期，每航次决策间隔至少 {audit['minimum_spacing_us']/1e6:g} 秒。特征源通过A2既有SHA校验，全部X/U/Y与源特征按 `(flight,t)` 一对一连接后核对；PACLD factual CSV也按该键连接，并验证原生目标时间戳。没有按CSV行序拼接不同来源。A2未来控制数组与已逐行验证X/U/Y的同一归档身份绑定，沿用既有ZOH和质量合同。

原A2无独立身份键的预测NPZ不直接作为误差对照；本轮重算LODO Ridge（17外层日期、其余16日期内部3折，alpha=[1,10,100,1000]），主控制保留原8控制联合选参口径，M0/M1保留原p/r双输出选参。M1仍含A2的四通道、50/100 ms共8个控制特征；这里只评分差动残差×滚转误差，不把M1优势归因于差动单通道。未来控制用同一合同对8个未来控制联合交叉拟合。每行预测都来自不含该日期的模型。详见 `folds.csv`。

U为前100 ms差动 `(right-left)/2` ZOH均值减决策前命令，归一化命令单位；它是闭环未来实际控制记录，**不是随机激励**。p为FRD滚转角速度，rad/s；Y使用观测起点，Y_hat使用因果Belief起点。冻结PACLD沿日志ZOH传播到同一原生目标时刻，没有未来观测更新。Decoder输出整机有效加速度/角加速度，并非孤立翼载荷；本诊断不引入新质量、惯量或力矩参考点标定。

`s_u={moments['scales']['s_u'][0]:.10g}`（归一化命令）；`s_p={moments['scales']['s_p']:.10g}` rad/s。两者为共同train窗口RMS。m=(U_res/s_u)(E_increment/s_p)无量纲；先窗口均值、再日期内航次等权、最后17日期等权。所有对照同一集合、同一s_p与加权口径；未来控制分别用自己的残差RMS。`E_endpoint=E_increment+E_initial`最大误差 {audit['endpoint_identity_max_error']:.3g} rad/s。

## A/B：残差与控制误差矩

{table(mt)}

主矩 **M={tot['moment_normal']['M']:.8f}**，M²={moments['L_moment']:.8f}。M_obs−M_model=M_error数值恒等成立。PACLD预测仅体现该观察性关联的一小部分：模型矩 {tot['moment_model']['M']:.6f}，观测矩 {tot['moment_obs']['M']:.6f}。不能据此声称真实因果控制增益偏小。M0矩 {tot['moment_M0']['M']:.6f}，M1矩 {tot['moment_M1']['M']:.6f}；样本外M1说明控制包含状态Ridge未解释的预测信息。

日期区间来自20,000次日期聚类bootstrap（固定seed=17），只作描述性区间，独立簇仅17个。主矩16/17日期为正，{int((flights.moment_normal>0).sum())}/{len(flights)}航次为正。逐日期有符号矩如下；完整逐航次见 `flight_moments.csv`。

{table(days[['date','moment_normal','moment_future50','moment_future100','moment_initial']])}

![各日期矩](control_residual_moment_v1/date_moments.png)

剔除任一日期后的主矩范围 [{min(influence['leave_one_date_out'].values()):.6f}, {max(influence['leave_one_date_out'].values()):.6f}]。仅作影响诊断，删除最高1%/5%绝对残差窗口后主矩分别为 {influence['tail'][0]['M']:.6f}/{influence['tail'][1]['M']:.6f}，下降约14%/37%；主要结论并非单个日期造成，但幅度依赖较大控制变化。主结果没有改筛选阈值或归一化。

初始Belief矩总体 {tot['moment_initial']['M']:.6f}，区间跨零，但各日期可达 [{tot['moment_initial']['date_min']:.6f}, {tot['moment_initial']['date_max']:.6f}]，大于部分日期传播矩。端点误差不能直接当作控制传播误差；采用增量误差消除了起点直接相加项，仍不等于消除初始状态对后续传播的影响。

主U RMS={res['raw']['normal']['rms']:.6f}，U_res RMS={res['residual']['normal']['rms']:.6f}。最大日期/航次残差平方能量占比为 {normal[normal.group=='date'].energy_share.max():.2%}/{normal[normal.group=='flight'].energy_share.max():.2%}。各日期/航次/模式分位数见 `residual_distributions.csv`，总体见 `residual_summary.json`。剩余线性相关绝对值最大的特征：

{table(selected_corr[['feature','correlation']])}

特征索引：0–2体轴速度，3–5 p/q/r，6–8重力方向，9频率，10–11相位sin/cos，12–23当前/前0.5秒控制历史，24–26率参考，27–30姿态参考，31空速，32–35模式标记，36–47可用性，48起模式one-hot。残差化不保证非线性独立，更不消除未观测反馈。

## C：未来控制负对照

未来250–300 / 250–350 ms矩分别为 {vals[1]:.6f}/{vals[2]:.6f}，绝对值为主矩的 {abs(vals[1]/vals[0]):.1%}/{abs(vals[2]/vals[0]):.1%}，分别16/17及14/17日期为负。未来控制不可能引起此前结束的响应；其量级与主矩同阶且符号相反，符合响应后反馈纠偏、自相关、持续参考的可能解释，无法在此分离这些机制。主/未来残差相关系数为 {res['residual_correlations'][0][1]:.6f}/{res['residual_correlations'][0][2]:.6f}。不能用符号相反或未来矩略小消除混杂风险。

![负对照](control_residual_moment_v1/negative_controls.png)

## D：批量噪声与梯度

固定随机种子、每种方案/B各10,000次：window_uniform为窗口均匀抽样（估计窗口均值，和目标M略有不同）；date_flight_balanced为按日期/航次层级概率独立抽样；date_flight_stratified为日期数量均衡、日期内航次数量均衡、航次内随机窗口。后两者期望对应目标M。日期/航次组成统计见 `batch_composition.csv` 和各表的mean_unique_flights。

{table(batch[['scheme','B','mean','bias_to_global','sd','sign_flip','M2_mean','mean_unique_flights']])}

层级随机抽样的单窗口方差分解：日期组成 {variance_parts['between_date']:.6f}、日期内航次组成 {variance_parts['between_flight']:.6f}、航次内窗口 {variance_parts['within_flight']:.6f}；独立抽样时各项除以B。该分解与分层抽样结果共同说明航次组成对估计的影响及无法靠均衡航次消除的窗口噪声。\n\n数值验证 E[M_batch²]=(E[M_batch])²+Var(M_batch)，误差<1e-14；层级采样无偏时才对应M²+Var，窗口均匀方案另有估计目标偏差。B=32噪声方差超过全局M²，平方矩均值约为全局目标的3倍，不能稳定代表系统偏差。B=256已大幅减少符号翻转；B=512是本轮测试中更保守的有效批量候选，仍存在方差项，**没有预注册可靠性门槛，不能声称证明了普适最小批量**。如进入受限试验，应保持日期/航次口径，使用至少512量级有效批量或跨批量估计，并单独验证估计器偏差。本轮未实现训练器。

梯度组在计算前固定，共 {audit['gradient_windows']} 窗口覆盖日期×四相位×残差符号可用格，随机分入4批。该队列为相位/符号覆盖性诊断样本，不是全train层级概率样本，不能把其M或平均梯度当作全train无偏估计。每批34个窗口；批内采用等窗口权重。每批FULL、M、M²使用相同身份与物理目标；FULL保持原生0.5秒窗口所有物理观测噪声加权目标，Moment使用指定0.2秒增量目标，因此比较的是两个真实目标，不是把FULL改成滚转MSE。Macro机制没有改动，不纳入FULL方向比较。微批量累加得到整批梯度，M²按2M∇M精确计算，并逐微批以autograd验证链式关系。

各诊断批次实际M与FULL值：\n\n{table(batch_checks)}\n\nGPU：{checks['device']}，精度策略 {checks['precision']}。所有参数逐元素验证未改变。重算factual与冻结CSV最大差 {max_parity:.3g} rad/s；autograd平方矩链式梯度最大差 {max_chain:.3g}。零向量余弦记为nan（未定义），不表示梯度非有限。各模块范数中位数如下；全部批次L2、参数RMS、最大绝对值及有限性见 `gradient_statistics.csv`。

{table(gs)}

跨批方向余弦（不是有效训练的证明）：

{table(st[st.objective.isin(['M','M2','M_stop','M2_stop'])])}

本次实际结果：M梯度的Fast coupling RMS中位数8.56e-8，Decoder为3.91e-3；停止Belief后分别为9.84e-9和4.06e-3。Decoder方向远比Fast coupling更容易被该目标推动，但参数尺度不同，不能把小范数等同于无训练价值。M²与FULL余弦中位数在Fast coupling/Decoder/Slow分别约−0.406/−0.248/−0.402；跨批M²余弦中位数分别约−0.059/−0.871/−0.673，存在方向冲突和批量不稳定。停止Belief后Fast coupling跨批余弦中位数约0.221但最小−0.352，Decoder仍约−0.877；detach没有解决总体方向噪声。四批M为−0.0432、−0.2241、0.4672、0.4792，不能以其中任一批认定训练可行。所有梯度有限；没有预设“极端”阈值，最大绝对分量与范数逐批公开，不据此声称完全没有优化风险。\n\nFast coupling的梯度可达：命令−执行器状态经coupling驱动Fast包络，后经Decoder映射滚转。equilibrium/relaxation、Slow、Decoder及Belief也可获得梯度。停止Moment初始Belief所有字段梯度后，未来Dynamics/Decoder方向仍可达，Encoder梯度应为零；这只阻止Moment利用初始Belief，不改变原FULL/Macro训练机制。

全train层级加权归一化残差均值为 {bias['weighted_normalized_residual_mean']:.6f}，故统一增量偏置对M的导数为 {bias['derivative_M_wrt_constant_increment_bias']:.6f} (rad/s)⁻¹；单靠统一偏置消除M需约 {bias['constant_increment_shift_to_cancel_global_moment']:.3f} rad/s增量，不能视为廉价的统一偏置解，但状态相关偏置路径仍未排除。\n\n这些模块都依赖状态和潜变量，M目标仅限制一个加权平均残差，未专门限制控制导数。Decoder输出偏置、状态依赖项、Slow/Fast状态修正都可能降低M；仅凭模块范数不能判断改变是否为正确控制传播。优化目标是M²，其梯度为2M∇M；当批量M翻转符号时，M²梯度也会翻转。因此判断与FULL冲突必须看M²而不只看M的余弦。FULL余弦与跨批余弦必须结合上表解释，单批较大梯度或小范数均不能独立认定有效/无效。当前证据是梯度可达与有限样本稳定性诊断，不是参数更新或闭环收益证明。

![批量与梯度](control_residual_moment_v1/batch_gradient_stability.png)

## 工况依赖

前飞速度按train三分位边界 {audit['speed_tertiles']} m/s分组；相位按Hall机械相位四等分。组内重做航次/日期等权，组间总体矩不等于简单平均（覆盖不同）。

{table(groups[['group','key','M','dates','flights','windows','positive_dates','negative_dates']])}

所有总体模式/速度/相位组的矩为正，没有发现分组总体正负抵消；但相位第3组（从0计数）仅9/17日期为正，M约0.037，远弱于第2组约0.207。尚无证据要求改用多组损失，却不能宣称统一M²解决每个局部工况。应区分总体分组符号与个别日期内的相反方向；逐组日期矩已保存。本轮不添加条件损失版本。若后续采用观察性约束，需保留这些条件诊断，不能把全局M²下降说成所有工况控制响应已解决；相位关联也不能当作相位独立因果作用。

## 四项判断与边界

- **A 信号存在性：VERIFIED。** 在给定Ridge条件特征下，PACLD误差与未解释控制变化存在正矩；不等于所有决策前状态均无法解释。
- **B 统计可靠性：SUPPORTED。** 大多数日期/航次同号，日期bootstrap不跨零；17簇限制推断，尾部窗口影响幅度，局部日期存在异号。
- **C 优化可行性：可达性VERIFIED，稳定训练价值UNRESOLVED。** 梯度可达并检查有限性；较大有效批量明显降低标量矩噪声，但本轮未验证512窗口的参数梯度方向；4×34覆盖性批次显示M²方向不稳，尚未证明优化能改善预测。模块梯度不能防止状态偏置路径绕过目标。
- **D 科学解释边界：强限制。** 未来控制负对照同阶、初始Belief日期污染和闭环反馈，使其不能作为准确控制增益监督。CONDITIONAL GO的条件是保留观察性解释、较大有效批量和FULL对照，不把本轮的梯度可达当作稳定收益承诺；仅可考虑受限等预算对照，本轮不启动。

最值得保留的替代监督方向是因果可用参考与动作、原生滚转增量的多步预测误差，以及具有明确外生来源/设计干预的局部响应证据；在没有外生干预前，M1结果只支持加入动作相关预测信息。

## 输出与复现

所有结果位于独立worktree `.worktrees/pacld-control-residual-moment-v1`。`audit.json`、`residual_summary.json`、`residual_distributions.csv`、`residual_feature_correlations.csv`、`date_moments.csv`、`flight_moments.csv`、`moments.json`、`batch_statistics.csv`、`batch_stratified.csv`、`batch_composition.csv`、`gradient_statistics.csv`、`gradient_stability.csv`、`gradient_checks.json`、`condition_moments.csv`、`conclusion.json`覆盖机器可读交付；`windows.csv`含身份、预测与残差。梯度向量NPZ仅本地保存，由脚本重算，不发布checkpoint或原始数据。

```bash
cd /home/zn/flap-system-identification/.worktrees/pacld-control-residual-moment-v1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:scripts
CRM_PY=/home/zn/anaconda3/envs/flap-train-gpu/bin/python
$CRM_PY scripts/pacld_crm_v1.py
$CRM_PY scripts/pacld_crm_gradients.py --microbatch 4
$CRM_PY scripts/pacld_crm_report.py
$CRM_PY scripts/verify_pacld_crm_v1.py
```

已执行完整train统计、136窗口GPU梯度、独立数值验证（`verification.json`: PASS），并检查三张图的可读性。未运行训练或MPC。\n\n复现依赖上文冻结本地源路径；GPU脚本会复用已完成批次向量，仅适用于本轮同一冻结配置。新诊断应另设输出目录，保留本轮产物。
'''
    (BASE/'control_residual_moment_feasibility_v1_zh.md').write_text(text)

if __name__=='__main__':
    import sys
    if not (OUT/'batch_stratified.csv').exists():supplements()
    if '--supplements-only' not in sys.argv:render()
