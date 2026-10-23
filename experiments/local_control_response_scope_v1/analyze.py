"""Train-first, outcome-blind scope selection. Run train, freeze, then validation.

Only explicitly named development Parquet files and allowed log headers are read.
Historical audit supplies unchanged history and exact ZOH comparison utilities.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import time

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from pyulog import ULog

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('old_audit', HERE.parent/'control_identifiability_audit_v1/audit.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
COORDS = ['motor', 'common', 'differential', 'rudder']


def coords(u):
    return np.stack([u[..., 0], (u[..., 1]+u[..., 2])/2,
                     (u[..., 2]-u[..., 1])/2, u[..., 3]], axis=-1)


def canonical_tape(t, u):
    """Lossless four-value ZOH reduction for statistics; no model code modified."""
    keep = np.r_[True, np.any(u[1:] != u[:-1], axis=1)]
    return t[keep], u[keep]


def integral_stats(t, u, threshold):
    """Exact statistics of a relative command difference on [0,H)."""
    dt = np.diff(t).astype(float)
    assert len(u) == len(dt) and np.all(dt > 0)
    weight = dt / dt.sum()
    mean = np.sum(u*weight[:, None], axis=0)
    rms = np.sqrt(np.sum(u*u*weight[:, None], axis=0))
    peak = np.max(np.abs(u), axis=0)
    longest = np.zeros(4)
    pos = np.zeros(4)
    neg = np.zeros(4)
    for c in range(4):
        for sign in [1, -1]:
            active = sign*u[:, c] >= threshold[c]
            duration = float(dt[active].sum())
            (pos if sign == 1 else neg)[c] = duration
            run = 0
            for d, ok in zip(dt, active):
                run = run+d if ok else 0
                longest[c] = max(longest[c], run)
    return mean, rms, peak, longest, pos, neg


def direction_flags(stats, cfg):
    mean, rms, peak, longest, pos, neg = stats
    th = np.asarray(cfg['change_thresholds'])
    energy = (rms/th)**2
    share = energy / max(float(energy.sum()), 1e-30)
    dominant = ((rms >= th) & (share >= cfg['direction_energy_fraction_min']) &
                (np.abs(mean) >= cfg['signed_mean_over_rms_min']*rms) &
                (longest >= cfg['minimum_contiguous_change_us']))
    quiet = bool(np.all(rms <= cfg['background_rms_limits']))
    return dominant, quiet, share


def cut_tape(native, start, end):
    t, u, _ = native
    ix = np.searchsorted(t, start, side='right')-1
    stop = np.searchsorted(t, end, side='left')
    if ix < 0:
        raise ValueError('missing initial command')
    tt = np.r_[start, t[ix+1:stop]]
    uu = u[ix:stop]
    tt, uu = canonical_tape(tt, uu)
    return np.r_[tt-start, end-start], uu


def contrast(a, b):
    ta, ua = a
    tb, ub = b
    t = np.unique(np.r_[ta, tb])
    return t, old.zoh(ta[:-1], ua, t[:-1])-old.zoh(tb[:-1], ub, t[:-1])


def load_data(data, split, cfg, raw_root):
    old.permitted(split)
    registry = yaml.safe_load((ROOT/cfg['registry']).read_text())['datasets'][cfg['dataset_id']]
    if old.sha(data/'dataset_manifest.json') != registry['manifest_sha256']:
        raise ValueError('registered manifest mismatch')
    manifest = json.loads((data/'dataset_manifest.json').read_text())
    names = ['flights.csv', 'flight_provenance.json', 'split_manifest.json', 'build_config.yaml']
    names += [f'{s}_{split}.parquet' for s in ['samples', 'windows', 'native_actuator_motors', 'native_actuator_servos']]
    hashes = {}
    for n in names:
        hashes[n] = old.sha(data/n)
        if hashes[n] != manifest['artifact_sha256'][n]:
            raise ValueError(f'artifact mismatch: {n}')
    f = pd.read_csv(data/'flights.csv')
    f = f[(f.split == split) & (f.sampling_group == cfg['sampling_group'])].copy()
    prov = {r['flight_id']: r for r in json.loads((data/'flight_provenance.json').read_text()) if r['flight_id'] in set(f.flight_id)}
    assign = {r['flight_id']:r['split'] for r in json.loads((data/'split_manifest.json').read_text())['assignments']}
    assert all(assign[x] == split for x in f.flight_id)
    columns = old.META+old.V+old.Q+old.W+old.U+['relative_flap_phase_rad', 'state_age_us', 'status_publication_us']
    s = old.load_parquet(data, 'samples', split, columns)
    s = s[s.flight_id.isin(f.flight_id)]
    groups = {fid:g.reset_index(drop=True) for fid,g in s.groupby('flight_id', sort=False)}
    m = old.load_parquet(data, 'native_actuator_motors', split)
    v = old.load_parquet(data, 'native_actuator_servos', split)
    md = dict(tuple(m[m.flight_id.isin(f.flight_id)].groupby('flight_id')))
    vd = dict(tuple(v[v.flight_id.isin(f.flight_id)].groupby('flight_id')))
    natives = {fid:old.tape(md[fid], vd[fid]) for fid in groups}
    w = old.load_parquet(data, 'windows', split)
    w = w[(w.history_us == cfg['history_us']) & (w.horizon_us == cfg['horizon_us']) & w.flight_id.isin(f.flight_id)]
    controllers = []
    for row in f.itertuples():
        p = raw_root/row.log_path
        # The allowlist is established from the authorized split before opening a raw file.
        assert assign[row.flight_id] == split
        if not p.is_file():
            controllers.append(dict(flight_id=row.flight_id, status='missing', controller_id='unknown'))
            continue
        h = old.sha(p)
        if h != row.sha256:
            raise ValueError(f'raw log hash mismatch {row.flight_id}')
        log = ULog(str(p), parse_header_only=True)
        params = {k:v for k,v in log.initial_parameters.items() if k.startswith(('FW_', 'FLAP_', 'SLOW_', 'CA_', 'PWM_MAIN_', 'ASPD_', 'NPFG_', 'TECS_'))}
        cid = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
        controllers.append(dict(flight_id=row.flight_id, status='header_verified', raw_path=str(p), raw_sha256=h,
                                controller_id=cid, parameters=params, initial_only=True))
    receipt = dict(dataset_id=cfg['dataset_id'], root=str(data), manifest_sha256=registry['manifest_sha256'],
                   artifact_sha256=hashes, partitions=[split], sampling_group=cfg['sampling_group'],
                   phase_contract=manifest['config']['phase'], sample_columns=columns,
                   test_artifacts_opened=False, target_columns_read=False, raw_log_read_mode='header only plus byte hash',
                   source_flights=len(f), registered_windows=len(w))
    return f, prov, groups, natives, w, controllers, receipt


def enrich(meta, hist, groups, prov, controllers, cfg):
    r = Rotation.from_quat(hist[:, :, 3:7].reshape(-1, 4)[:, [1, 2, 3, 0]])
    euler = r.as_euler('xyz', degrees=True).reshape(len(hist), -1, 3)
    vb = r.inv().apply(hist[:, :, :3].reshape(-1, 3)).reshape(len(hist), -1, 3)
    speed = np.linalg.norm(hist[:, :, :3], axis=2)
    rate = np.linalg.norm(hist[:, :, 7:10], axis=2)
    for key, val in dict(roll_max=np.abs(euler[:, :, 0]).max(1), pitch_max=np.abs(euler[:, :, 1]).max(1),
                         rate_rms=np.sqrt((rate**2).mean(1)), rate_initial=rate[:, -1],
                         speed_min=speed.min(1), speed_max=speed.max(1), body_forward_min=vb[:, :, 0].min(1),
                         vertical_max=np.abs(hist[:, :, 2]).max(1), frequency_min=hist[:, :, 10].min(1),
                         frequency_max=hist[:, :, 10].max(1), initial_roll=euler[:, -1, 0], initial_pitch=euler[:, -1, 1]).items():
        meta[key] = val
    near = (hist[:, :, 11] >= .98) | (hist[:, :, 11] <= .02) | (np.abs(hist[:, :, 12:15]) >= .98).any(2)
    meta['history_near_limit_fraction'] = near.mean(1)
    for fid, g in meta.groupby('flight_id'):
        s = groups[fid]
        ix = g.origin_sample.to_numpy()
        meta.loc[g.index, 'phase_relative_origin'] = s.relative_flap_phase_rad.to_numpy()[ix]
        meta.loc[g.index, 'origin_state_age_us'] = s.state_age_us.to_numpy()[ix]
        meta.loc[g.index, 'date'] = prov[fid]['flight_date']
    cid = {r['flight_id']:r['controller_id'] for r in controllers}
    meta['controller_id'] = meta.flight_id.map(cid)
    b = cfg['common_history_bounds']
    common = ((meta.speed_min >= b['ground_speed_min']) & (meta.speed_max <= b['ground_speed_max']) &
              (meta.body_forward_min >= b['body_forward_min']) & (meta.frequency_min >= b['frequency_min']) &
              (meta.frequency_max <= b['frequency_max']) & (meta.history_near_limit_fraction <= b['near_limit_fraction_max']))
    for name, c in cfg['regimes'].items():
        meta[name] = (common & (meta.nav_state == c['mode']) & (meta.roll_max <= c['roll_max_deg']) &
                      (meta.pitch_max <= c['pitch_max_deg']) & (meta.rate_rms <= c['rate_rms_max']) &
                      (meta.rate_initial <= c['rate_rms_max']) & (meta.vertical_max <= c['vertical_max']))
    return meta


def nonoverlap_windows(meta):
    keep = []
    for _, g in meta.groupby('log_sha', sort=True):
        end = -1
        for row in g.sort_values(['decision_us', 'window_id']).itertuples():
            if row.history_start_us > end:
                keep.append(row.Index)
                end = row.pair_support_end_us
    return keep


def union_seconds(meta):
    total = 0
    for _, g in meta.groupby('log_sha'):
        end = -1
        for row in g.sort_values('decision_us').itertuples():
            lo, hi = row.decision_us, row.command_end_us
            total += max(0, hi-max(lo, end))
            end = max(end, hi)
    return total*1e-6


def graph_counts(p, meta):
    parent = {}
    def root(a):
        parent.setdefault(a,a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    nodes = p[['i','j']].to_numpy().ravel().astype(int)
    logs = meta.log_sha.to_numpy()
    for a,b in p[['i','j']].itertuples(index=False, name=None):
        parent[root(logs[a])] = root(logs[b])
    return dict(unique_nodes=len(set(nodes)), shared_node_uses=len(nodes)-len(set(nodes)),
                flights=len(parent), flight_components=len({root(a) for a in parent}),
                max_node_degree=int(pd.Series(nodes).value_counts().max()) if len(nodes) else 0)


def event_table(meta, natives, cfg):
    rows = []
    for fid, g in meta.groupby('flight_id', sort=False):
        for row in g.itertuples():
            t,u = cut_tape(natives[fid], row.decision_us, row.command_end_us)
            cu = coords(u)
            stats = integral_stats(t, cu-cu[0], cfg['change_thresholds'])
            dom, quiet, share = direction_flags(stats,cfg)
            out = dict(i=row.Index, quiet=quiet, future_near_limit_fraction=float(np.dot(np.diff(t)/cfg['horizon_us'],
                       (u[:,0] >= .98) | (u[:,0] <= .02) | (np.abs(u[:,1:]) >= .98).any(1))), canonical_segments=len(u))
            for c,ch in enumerate(COORDS):
                for label, value in zip(['mean','rms','peak','longest_us','positive_us','negative_us'], stats):
                    out[f'{ch}_{label}'] = value[c]
                out[f'{ch}_dominant'] = bool(dom[c])
                out[f'{ch}_energy_share'] = share[c]
            raw = integral_stats(t, u-u[0], cfg['change_thresholds'])
            for c,ch in enumerate(old.CHANNELS):
                out[f'raw_{ch}_mean'] = raw[0][c]
                out[f'raw_{ch}_rms'] = raw[1][c]
            rows.append(out)
    return pd.DataFrame(rows).set_index('i').reindex(meta.index)


def match(meta, hist, feature, natives, cfg):
    records = []
    search_count = 0
    modes = meta.groupby('nav_state').groups
    for mode, index in modes.items():
        ids = np.asarray(index)
        tree = cKDTree(feature[ids])
        for start in range(0, len(ids), 256):
            lists = tree.query_ball_point(feature[ids[start:start+256]], 1.000002, p=np.inf, workers=2)
            aa,bb = [],[]
            for i, js in zip(ids[start:start+256], lists):
                j = ids[np.asarray(js,int)]
                j = j[j > i]
                aa.extend([i]*len(j)); bb.extend(j)
            aa,bb = np.asarray(aa,int),np.asarray(bb,int)
            good = ((meta.log_sha.to_numpy()[aa] != meta.log_sha.to_numpy()[bb]) &
                    (meta.hardware_uuid.to_numpy()[aa] == meta.hardware_uuid.to_numpy()[bb]) &
                    (meta.hardware_uuid.to_numpy()[aa] != '') &
                    (np.abs(meta.ratio.to_numpy()[aa]-meta.ratio.to_numpy()[bb]) <= cfg['match_max_ratio_difference']))
            aa,bb = aa[good],bb[good]
            search_count += len(aa)
            for st in range(0,len(aa),2048):
                a,b = aa[st:st+2048],bb[st:st+2048]
                err = old.history_errors(hist[a],hist[b])
                ok = old.caliper_mask(err,cfg['calipers']['primary'])
                if ok.any():
                    records.append(np.c_[a[ok],b[ok],err[ok]])
        print('matched mode', mode, 'search candidates', search_count, flush=True)
    cols = ['i','j','v_rms','attitude_rms_deg','rate_rms','frequency_rms']+['hist_'+x for x in old.CHANNELS]
    cols += ['v_initial','attitude_initial_deg','rate_initial','frequency_initial']+['initial_'+x for x in old.CHANNELS]
    p = pd.DataFrame(np.concatenate(records) if records else np.empty((0,len(cols))),columns=cols)
    p[['i','j']] = p[['i','j']].astype(int)
    scales = np.tile([.75,15,.75,.5]+[.05]*4,2)
    p['score'] = np.sqrt(np.mean((p[cols[2:]].to_numpy()/scales)**2,axis=1))
    p['strict'] = old.caliper_mask(p[cols[2:]].to_numpy(),cfg['calipers']['strict'])
    extra = []
    for row in p.itertuples():
        a,b = meta.loc[row.i],meta.loc[row.j]
        t,u = contrast(cut_tape(natives[a.flight_id],a.decision_us,a.command_end_us),
                       cut_tape(natives[b.flight_id],b.decision_us,b.command_end_us))
        stats = integral_stats(t,coords(u),cfg['change_thresholds'])
        dom,quiet,share = direction_flags(stats,cfg)
        hd = coords(hist[row.i,:,11:])-coords(hist[row.j,:,11:])
        hgap = np.sqrt(np.mean(hd**2,axis=0))
        initial = np.abs(hd[-1])
        strong = stats[1] >= cfg['future_to_history_ratio_min']*np.maximum(hgap,initial)
        r = dict(background_pair=quiet, same_firmware=a.firmware==b.firmware,
                 same_controller=(a.controller_id==b.controller_id and a.controller_id!='unknown'),
                 same_date=a.date==b.date, any_review=bool(a.history_review or b.history_review))
        for c,ch in enumerate(COORDS):
            r[ch] = bool(dom[c] and strong[c])
            r[ch+'_unstrengthened'] = bool(dom[c])
            for label,value in zip(['mean','rms','peak','longest_us','positive_us','negative_us'],stats):
                r[f'difference_{ch}_{label}'] = value[c]
            r[f'difference_{ch}_energy_share'] = share[c]
        extra.append(r)
    if not extra:
        raise ValueError('no history matches; report empty support explicitly')
    return pd.concat([p,pd.DataFrame(extra)],axis=1), search_count


def summarize(meta,events,p,cfg,out):
    coverage, distributions, pairrows, selections = [],[],[],[]
    byflight=[]
    for regime in cfg['regimes']:
        g = meta[meta[regime]]
        ix = g.index
        nidx = nonoverlap_windows(g)
        coverage.append(dict(regime=regime,windows=len(g),nonoverlap_windows=len(nidx),flights=g.flight_id.nunique(),
                             dates=g.date.nunique(),future_union_s=union_seconds(g),background_windows=int(events.loc[ix,'quiet'].sum()),
                             background_nonoverlap=len(nonoverlap_windows(g[events.loc[ix,'quiet']]))))
        for fid, fg in g.groupby('flight_id'):
            byflight.append(dict(regime=regime,flight_id=fid,date=fg.date.iloc[0],firmware=fg.firmware.iloc[0],
                                 controller_id=fg.controller_id.iloc[0],windows=len(fg),nonoverlap_windows=len(nonoverlap_windows(fg)),
                                 future_union_s=union_seconds(fg)))
        for ch in COORDS:
            ev=events.loc[ix]
            dom=ev[ch+'_dominant']
            selected=g[dom]
            for measure in [ch+'_rms',ch+'_mean',ch+'_longest_us','raw_'+old.CHANNELS[COORDS.index(ch)]+'_rms']:
                distributions.append(dict(regime=regime,coordinate=ch,measure=measure,**old.qstats(ev[measure])))
            coverage.append(dict(regime=regime,coordinate=ch,windows=len(selected),nonoverlap_windows=len(nonoverlap_windows(selected)),
                                 flights=selected.flight_id.nunique(),dates=selected.date.nunique(),future_union_s=union_seconds(selected),
                                 positive=int((ev.loc[dom,ch+'_mean']>0).sum()),negative=int((ev.loc[dom,ch+'_mean']<0).sum())))
        pg = p[meta[regime].to_numpy()[p.i] & meta[regime].to_numpy()[p.j]]
        for ch in COORDS+['background_pair']:
            sub=pg[pg[ch]]
            accepted=old.disjoint_pairs(sub,meta)
            flightdis=old.disjoint_pairs(sub,meta,True)
            selected=sub.loc[accepted]
            quiet_a=events.quiet.to_numpy()[selected.i]
            quiet_b=events.quiet.to_numpy()[selected.j]
            pairrows.append(dict(regime=regime,coordinate=ch,raw_pairs=len(sub),nonoverlap_pairs=len(accepted),
                                 flight_disjoint_pairs=len(flightdis),**graph_counts(selected,meta),
                                 raw_shared_node_uses=graph_counts(sub,meta)['shared_node_uses'],
                                 strict_nonoverlap_pairs=len(old.disjoint_pairs(sub[sub.strict],meta)),
                                 same_firmware_nonoverlap=len(old.disjoint_pairs(sub[sub.same_firmware],meta)),
                                 same_controller_nonoverlap=len(old.disjoint_pairs(sub[sub.same_controller],meta)),
                                 same_firmware_controller_nonoverlap=len(old.disjoint_pairs(sub[sub.same_firmware & sub.same_controller],meta)),
                                 cross_date_nonoverlap=len(old.disjoint_pairs(sub[~sub.same_date],meta)),
                                 one_quiet_member=int((quiet_a ^ quiet_b).sum()),both_quiet=int((quiet_a & quiet_b).sum()),
                                 v_rms_median=float(selected.v_rms.median()) if len(selected) else None,
                                 rate_rms_median=float(selected.rate_rms.median()) if len(selected) else None,
                                 attitude_rms_deg_median=float(selected.attitude_rms_deg.median()) if len(selected) else None))
            for idx in accepted:
                selections.append(dict(regime=regime,coordinate=ch,pair_id=idx,flight_disjoint=idx in flightdis))
    pd.DataFrame(coverage).to_csv(out/'coverage.csv',index=False)
    pd.DataFrame(distributions).to_csv(out/'command_distributions.csv',index=False)
    pd.DataFrame(byflight).to_csv(out/'coverage_per_flight.csv',index=False)
    result=pd.DataFrame(pairrows)
    result.to_csv(out/'pair_support.csv',index=False)
    pd.DataFrame(selections,columns=['regime','coordinate','pair_id','flight_disjoint']).to_csv(out/'selected_pairs.csv',index=False)
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--split',required=True,choices=['train','validation'])
    ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--raw-root',type=Path,default=Path('/home/zn/QgcLogs'))
    ap.add_argument('--output-root',type=Path,default=HERE/'results')
    args=ap.parse_args()
    cfg=json.loads((HERE/'config.json').read_text())
    lock=args.output_root/'frozen_selection.json'
    if args.split=='validation':
        frozen=json.loads(lock.read_text())
        assert frozen['config_sha256']==old.sha(HERE/'config.json')
        assert frozen['analysis_sha256']==old.sha(Path(__file__))
    out=args.output_root/args.split
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'refuse overwrite {out}')
    out.mkdir(parents=True,exist_ok=True)
    start=time.time()
    f,prov,groups,natives,w,controllers,receipt=load_data(args.data_root,args.split,cfg,args.raw_root)
    old.write_json(out/'source_receipt.json',receipt)
    old.write_json(out/'controller_headers.json',controllers)
    meta,hist,feature,eligibility=old.build_histories(cfg,f,prov,groups,natives,w,cfg['horizon_us'])
    eligibility.to_csv(out/'eligibility.csv',index=False)
    meta=enrich(meta,hist,groups,prov,controllers,cfg)
    print('eligible',len(meta),'regimes',meta[list(cfg['regimes'])].sum().to_dict(),flush=True)
    # Preserve full eligibility diagnostics, then restrict the pair search to candidate regimes.
    meta.to_csv(out/'all_eligible_metadata.csv.gz',index=False)
    chosen=meta[list(cfg['regimes'])].any(axis=1).to_numpy()
    meta=meta.loc[chosen].reset_index(drop=True)
    hist,feature=hist[chosen],feature[chosen]
    meta['candidate_id']=np.arange(len(meta))
    events=event_table(meta,natives,cfg)
    events.to_csv(out/'command_events.csv.gz',index=True,index_label='candidate_id')
    print('command features',len(events),flush=True)
    p,nsearch=match(meta,hist,feature,natives,cfg)
    p.to_csv(out/'all_history_pairs.csv.gz',index=True,index_label='pair_id')
    meta.to_csv(out/'candidate_index.csv.gz',index=False)
    summary=summarize(meta,events,p,cfg,out)
    if args.split=='train':
        pool=summary[summary.coordinate.isin(COORDS)].copy()
        retained=[]
        for mode in ['AUTO','STAB']:
            for ch in COORDS:
                stable=pool[(pool.regime==mode+'_stable') & (pool.coordinate==ch)].iloc[0]
                broad=pool[(pool.regime==mode+'_forward') & (pool.coordinate==ch)].iloc[0]
                retained.append(stable if stable.nonoverlap_pairs>=0.8*broad.nonoverlap_pairs else broad)
        ranked=pd.DataFrame(retained).sort_values(
            ['nonoverlap_pairs','flight_disjoint_pairs','flights','regime','coordinate'],ascending=[False,False,False,True,True])
        backup=ranked[ranked.coordinate!=ranked.iloc[0].coordinate].iloc[0]
        old.write_json(lock,dict(config_sha256=old.sha(HERE/'config.json'),analysis_sha256=old.sha(Path(__file__)),
                                 created_unix=time.time(),selection_data='train only',
                                 ranking=ranked[['regime','coordinate','nonoverlap_pairs','flight_disjoint_pairs','flights']].to_dict('records'),
                                 recommendation=ranked.iloc[0][['regime','coordinate']].to_dict(),
                                 backup=backup[['regime','coordinate']].to_dict()))
    old.write_json(out/'execution.json',dict(argv=sys.argv,seconds=time.time()-start,python=sys.executable,
                    config_sha256=old.sha(HERE/'config.json'),analysis_sha256=old.sha(Path(__file__)),
                    old_audit_sha256=old.sha(HERE.parent/'control_identifiability_audit_v1/audit.py'),
                    search_radius_candidates=nsearch,history_pairs=len(p),model_loaded=False,test_opened=False))
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__':
    main()
