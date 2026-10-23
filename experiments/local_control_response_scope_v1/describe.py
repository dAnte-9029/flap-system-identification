"""Descriptive tables/figures after the train-only rule is frozen; no reselection."""
from pathlib import Path
import argparse
import importlib.util
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('scope',HERE/'analyze.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--results',type=Path,default=HERE/'results')
    ap.add_argument('--data-root',type=Path,required=True)
    args=ap.parse_args()
    out=args.results/'description'
    out.mkdir(exist_ok=True)
    figdir=out/'figures'
    figdir.mkdir(exist_ok=True)
    cfg=json.loads((HERE/'config.json').read_text())
    frozen=json.loads((args.results/'frozen_selection.json').read_text())
    assert frozen['config_sha256']==m.old.sha(HERE/'config.json')
    regime=frozen['recommendation']['regime']
    direction=frozen['recommendation']['coordinate']
    parts={}
    perflight=[]; pairdetail=[]; features=[]; relationships=[]; shapes=[]; pairwave=[]
    backgrounds=[]; background_index=[]
    joint=[]
    for split in ['train','validation']:
        root=args.results/split
        meta=pd.read_csv(root/'candidate_index.csv.gz')
        events=pd.read_csv(root/'command_events.csv.gz',index_col='candidate_id')
        pairs=pd.read_csv(root/'all_history_pairs.csv.gz',index_col='pair_id')
        selection=pd.read_csv(root/'selected_pairs.csv')
        coverage=pd.read_csv(root/'coverage.csv')
        support=pd.read_csv(root/'pair_support.csv')
        parts[split]=(meta,events,pairs,selection,coverage,support)
        pg=pairs[meta[regime].to_numpy()[pairs.i]&meta[regime].to_numpy()[pairs.j]]
        qa=events.quiet.to_numpy()[pg.i];qb=events.quiet.to_numpy()[pg.j]
        for label,mask in [('both_future_quiet',qa&qb),('differential_vs_quiet',pg.differential&(qa^qb)),
                           ('motor_vs_quiet',pg.motor&(qa^qb)),('small_pair_and_both_quiet',pg.background_pair&qa&qb)]:
            sub=pg[mask];ids=m.old.disjoint_pairs(sub,meta)
            backgrounds.append(dict(split=split,regime=regime,cohort=label,raw_pairs=len(sub),
                               nonoverlap_pairs=len(ids),flight_disjoint_pairs=len(m.old.disjoint_pairs(sub,meta,True)),
                               **m.graph_counts(sub.loc[ids],meta)))
            background_index.extend(dict(split=split,regime=regime,cohort=label,pair_id=i) for i in ids)
        for reg in cfg['regimes']:
            g=meta[meta[reg]]
            nonoverlap=m.nonoverlap_windows(g)
            rawcols=['raw_'+ch+'_mean' for ch in m.old.CHANNELS]
            corr=events.loc[nonoverlap,rawcols].corr()
            for ci,ch in enumerate(m.old.CHANNELS):
                joint.append(dict(split=split,regime=reg,channel=ch,windows=len(nonoverlap),
                                  **dict(zip(m.old.CHANNELS,corr.iloc[ci].tolist()))))
            for field in ['history_speed','body_forward_min','initial_roll','initial_pitch','rate_rms',
                          'history_frequency','history_motor','history_left_elevon','history_right_elevon','history_rudder',
                          'origin_state_age_us','phase_relative_origin','history_near_limit_fraction']:
                features.append(dict(split=split,regime=reg,feature=field,**m.old.qstats(g[field])))
            for ch in m.COORDS:
                chosen=g[events.loc[g.index,ch+'_dominant']]
                keep=m.nonoverlap_windows(chosen)
                for fid,fg in chosen.groupby('flight_id'):
                    e=events.loc[fg.index]
                    ne=e.loc[e.index.intersection(keep)]
                    perflight.append(dict(split=split,regime=reg,direction=ch,flight_id=fid,date=fg.date.iloc[0],
                       windows=len(e),nonoverlap_windows=len(ne),positive=int((ne[ch+'_mean']>0).sum()),
                       negative=int((ne[ch+'_mean']<0).sum()),rms_p10=ne[ch+'_rms'].quantile(.1),
                       rms_p50=ne[ch+'_rms'].median(),rms_p90=ne[ch+'_rms'].quantile(.9),
                       dwell_p10=ne[ch+'_longest_us'].quantile(.1)*1e-6,dwell_p90=ne[ch+'_longest_us'].quantile(.9)*1e-6))
                for label,mask in [('all',np.ones(len(g),bool)),('speed_4_6',g.history_speed<6),
                                   ('speed_6_8',(g.history_speed>=6)&(g.history_speed<8)),('speed_8_12',g.history_speed>=8),
                                   ('rate_le_06',g.rate_rms<=.6),('rate_gt_06',g.rate_rms>.6)]:
                    ids=g.index[np.asarray(mask)]
                    relationships.append(dict(split=split,regime=reg,direction=ch,state_bin=label,windows=len(ids),
                          dominant=int(events.loc[ids,ch+'_dominant'].sum()),
                          rms_median=events.loc[ids,ch+'_rms'].median(),signed_mean_median=events.loc[ids,ch+'_mean'].median()))
            for ch in m.COORDS+['background_pair']:
                selected=selection[(selection.regime==reg)&(selection.coordinate==ch)]
                pp=pairs.loc[selected.pair_id]
                for row in pp.itertuples():
                    a,b=meta.loc[row.i],meta.loc[row.j]
                    pairdetail.append(dict(split=split,regime=reg,coordinate=ch,pair_id=row.Index,
                        window_a=a.window_id,window_b=b.window_id,flight_a=a.flight_id,flight_b=b.flight_id,
                        date_a=a.date,date_b=b.date,decision_a_us=a.decision_us,decision_b_us=b.decision_us,
                        support_a_start_us=a.history_start_us,support_a_end_us=a.pair_support_end_us,
                        support_b_start_us=b.history_start_us,support_b_end_us=b.pair_support_end_us,
                        quiet_a=bool(events.loc[row.i,'quiet']),quiet_b=bool(events.loc[row.j,'quiet']),
                        same_firmware=row.same_firmware,same_controller=row.same_controller,score=row.score,
                        flight_disjoint=bool(selected.set_index('pair_id').loc[row.Index,'flight_disjoint']),
                        v_rms=row.v_rms,rate_rms=row.rate_rms,attitude_rms_deg=row.attitude_rms_deg,
                        direction_rms=getattr(row,'difference_'+ch+'_rms',np.nan),
                        direction_dwell_s=getattr(row,'difference_'+ch+'_longest_us',np.nan)*1e-6))
        # Shape descriptions use a fixed nonoverlap subset of the recommended regime.
        # Native files have already been hash verified by the analysis; verify again on read.
        receipt=json.loads((root/'source_receipt.json').read_text())
        streams={}
        for stem in ['native_actuator_motors','native_actuator_servos']:
            name=f'{stem}_{split}.parquet'
            assert m.old.sha(args.data_root/name)==receipt['artifact_sha256'][name]
            streams[stem]=m.old.load_parquet(args.data_root,stem,split)
        mm=dict(tuple(streams['native_actuator_motors'].groupby('flight_id')))
        ss=dict(tuple(streams['native_actuator_servos'].groupby('flight_id')))
        ids=m.nonoverlap_windows(meta[meta[regime]])
        native={fid:m.old.tape(mm[fid],ss[fid]) for fid in meta.loc[ids,'flight_id'].unique()}
        bins=np.arange(0,500001,100000)
        for i in ids:
            row=meta.loc[i]
            t,u=m.cut_tape(native[row.flight_id],row.decision_us,row.command_end_us)
            c=m.coords(u); delta=c-c[0]
            # Add fixed bin boundaries, integrate exactly; do not interpolate commands.
            tt=np.unique(np.r_[t,bins]);dd=m.old.zoh(t[:-1],delta,tt[:-1]);dt=np.diff(tt)
            for k,ch in enumerate(m.COORDS):
                rec=dict(split=split,candidate_id=i,flight_id=row.flight_id,direction=ch,
                         dominant=bool(events.loc[i,ch+'_dominant']),sign=float(np.sign(events.loc[i,ch+'_mean'])),
                         rms=events.loc[i,ch+'_rms'],longest_s=events.loc[i,ch+'_longest_us']*1e-6,
                         positive_s=events.loc[i,ch+'_positive_us']*1e-6,negative_s=events.loc[i,ch+'_negative_us']*1e-6,
                         total_variation=float(np.abs(np.diff(c[:,k])).sum()),net_change=float(delta[-1,k]))
                for b in range(5):
                    sel=(tt[:-1]>=bins[b])&(tt[:-1]<bins[b+1])
                    rec[f'mean_{b*100}_{(b+1)*100}ms']=float(np.sum(dd[sel,k]*dt[sel])/100000)
                shapes.append(rec)
        # Exact tapes for score-first, low/mid/high amplitude pairs; deterministic IDs.
        chosen=selection[(selection.regime==regime)&(selection.coordinate==direction)]
        pp=pairs.loc[chosen.pair_id].sort_values('difference_'+direction+'_rms')
        picks=sorted(set(pp.iloc[np.unique(np.linspace(0,len(pp)-1,min(3,len(pp))).astype(int))].index)) if len(pp) else []
        for pid in picks:
            row=pairs.loc[pid]
            for side,i in [('A',int(row.i)),('B',int(row.j))]:
                win=meta.loc[i]
                if win.flight_id not in native:
                    native[win.flight_id]=m.old.tape(mm[win.flight_id],ss[win.flight_id])
                t,u=m.cut_tape(native[win.flight_id],win.decision_us,win.command_end_us)
                for ti,ui in zip(t[:-1],m.coords(u)):
                    pairwave.append(dict(split=split,pair_id=pid,side=side,time_s=ti*1e-6,**dict(zip(m.COORDS,ui))))
    for name,rows in [('direction_per_flight',perflight),('selected_pair_index',pairdetail),
                      ('history_distributions',features),('control_state_relationship',relationships),
                      ('nonoverlap_shapes',shapes),('example_tapes',pairwave),
                      ('background_comparability',backgrounds),('background_index',background_index),
                      ('joint_delta_correlations',joint)]:
        pd.DataFrame(rows).to_csv(out/(name+'.csv.gz' if name=='nonoverlap_shapes' else name+'.csv'),index=False)
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,split in zip(axs,['train','validation']):
        c=parts[split][4];c=c[c.coordinate.isna()]
        x=np.arange(len(c));ax.bar(x-.18,c.windows,.36,label='overlapping windows')
        ax.bar(x+.18,c.nonoverlap_windows,.36,label='disjoint full support')
        ax.set_xticks(x,c.regime,rotation=20);ax.set_yscale('log');ax.set_title(split);ax.legend(fontsize=8)
        ax.set_ylabel('Count (log scale)')
    fig.tight_layout();fig.savefig(figdir/'regime_coverage.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,split in zip(axs,['train','validation']):
        s=parts[split][5];s=s[s.regime==regime];x=np.arange(len(s))
        for offset,col,label in [(-.25,'nonoverlap_pairs','support disjoint'),(0,'flight_disjoint_pairs','flight disjoint'),(.25,'strict_nonoverlap_pairs','strict history')]:
            ax.bar(x+offset,s[col],.25,label=label)
        ax.set_xticks(x,s.coordinate,rotation=20);ax.set_ylabel('Pairs');ax.set_title(split+' / '+regime);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(figdir/'pair_support.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(11,7))
    for ax,ch in zip(axs.flat,m.old.CHANNELS):
        meta,ev=parts['train'][:2]
        for reg in ['STAB_stable','AUTO_stable']:
            ids=m.nonoverlap_windows(meta[meta[reg]])
            ax.hist(ev.loc[ids,'raw_'+ch+'_rms'],bins=np.linspace(0,.5,65),density=True,histtype='step',label=reg)
        ax.set(title=ch,xlabel='500 ms RMS change from decision command',ylabel='Density');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(figdir/'control_change_distributions.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5))
    meta,ev=parts['train'][:2];ids=m.nonoverlap_windows(meta[meta[regime]])
    h=ax.hexbin(ev.loc[ids,'raw_left_elevon_mean'],ev.loc[ids,'raw_right_elevon_mean'],gridsize=45,bins='log',mincnt=1)
    ax.set(xlabel='Mean left command change',ylabel='Mean right command change',title='Train '+regime+' / nonoverlap histories');fig.colorbar(h,ax=ax,label='log count')
    fig.tight_layout();fig.savefig(figdir/'joint_left_right.png',dpi=160);plt.close(fig)
    tape=pd.DataFrame(pairwave)
    fig,axs=plt.subplots(2,3,figsize=(13,6),squeeze=False)
    for row,split in enumerate(['train','validation']):
        ids=sorted(tape.loc[tape.split==split,'pair_id'].unique()) if len(tape) else []
        for col,ax in enumerate(axs[row]):
            if col>=len(ids):ax.axis('off');continue
            pid=ids[col]
            for side,g in tape[(tape.split==split)&(tape.pair_id==pid)].groupby('side'):
                ax.step(np.r_[g.time_s,.5],np.r_[g[direction],g[direction].iloc[-1]],where='post',label=side)
            ax.set(title=f'{split} pair {pid}',xlabel='Time after decision / s',ylabel=direction+' command');ax.legend()
    fig.tight_layout();fig.savefig(figdir/'example_command_shapes.png',dpi=160);plt.close(fig)
    # Check the inherited checkpoint references, without loading a network or selecting weights.
    cm=ROOT/'experiments/control_event_canonicalization_v1/results/integrated/manifest.json'
    ck=json.loads(cm.read_text())['checkpoints']
    assets=[]
    for name,h in ck.items():
        candidates=[ROOT/name,Path('/home/zn/flap-system-identification/.worktrees/control-event-canonicalization-v1')/name]
        found=next((p for p in candidates if p.exists()),None)
        actual=m.old.sha(found) if found else None
        if found:assert actual==h
        assets.append(dict(path=str(found) if found else name,registered_sha256=h,actual_sha256=actual,loaded=False))
    m.old.write_json(out/'asset_receipt.json',dict(checkpoints=assets,reference_manifest_sha256=m.old.sha(cm),
                  source_reports={p.name:m.old.sha(p) for p in (HERE/'sources').glob('*')},
                  core_parquet_recovery='local registered files; no ZIP reconstruction needed',
                  drive_stage5_zip=dict(id='17MR-fYfkIxgq8g_osfIEl9xuA9Zom6cM',bytes=81190672,listed=True,downloaded=False)))


if __name__=='__main__':main()
