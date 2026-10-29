"""Read-only adapter for frozen September/October velocity_rate_multistep_v1.

Uses accepted main_50hz and sparse_10hz windows only for identities/boundaries, native sensor tables
for events. No interpolation-created targets and no test partition entry point.
"""
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import pandas as pd
import torch
import yaml
from system_identification.models.pahbd import State,Observation,ControlTape,Window

TOPICS=('vehicle_local_position','vehicle_attitude','vehicle_angular_velocity',
        'wing_phase','flap_frequency','actuator_motors','actuator_servos')
FIELDS={'vehicle_local_position':['x','y','z','vx','vy','vz'],
        'vehicle_attitude':[f'q[{i}]' for i in range(4)],
        'vehicle_angular_velocity':[f'xyz[{i}]' for i in range(3)]}
DEFAULT_REGISTRY=str(Path(__file__).resolve().parents[3]/'configs/data/velocity_rate_dataset_registry.yaml')
DEFAULT_ROOT='/home/zn/flap-system-identification/dataset/velocity_rate_multistep_v1'


class NativeWindows:
    def __init__(self,root=DEFAULT_ROOT,registry=DEFAULT_REGISTRY,split='train',
                 history_us=500000,horizon_us=(200000,500000,1000000,2000000),dtype=torch.float64,device='cpu',
                 phase_policy='shared_or_average',reference_contract=None,sampling_groups=('main_50hz','sparse_10hz')):
        if split not in ('train','validation'):raise ValueError('sealed test is not an allowed split')
        if phase_policy not in ('shared_or_average','average'):raise ValueError('invalid phase policy')
        self.phase_policy=phase_policy
        self.phase_coefficients=(3.6354511983566393,23.393570068972757,-5.126725395181777)
        self.root=Path(root);self.split=split;self.dtype=dtype;self.device=device
        entry=yaml.safe_load(Path(registry).read_text())['datasets']['velocity_rate_multistep_v1']
        # One identity check at construction; never hash/open sealed artifacts.
        self.manifest_sha256=hashlib.sha256((self.root/'dataset_manifest.json').read_bytes()).hexdigest()
        if self.manifest_sha256!=entry['manifest_sha256']:raise ValueError('frozen manifest identity mismatch')
        self.contract=yaml.safe_load((self.root/'build_config.yaml').read_text())
        if self.contract['dataset_id']!='velocity_rate_multistep_v1':raise ValueError('wrong dataset family')
        self.provenance={m['flight_id']:m for m in json.loads((self.root/'flight_provenance.json').read_text()) if m['split']==split}
        w=pd.read_parquet(self.root/f'windows_{split}.parquet',filters=[('sampling_group','in',list(sampling_groups)),('history_us','==',history_us),('horizon_us','in',[horizon_us] if isinstance(horizon_us,int) else list(horizon_us))])
        self.windows=w[(w.sampling_group.isin(sampling_groups))&(w.history_us==history_us)&(w.horizon_us.isin([horizon_us] if isinstance(horizon_us,int) else horizon_us))
                       &w.firmware_verified&~w.provenance_review].reset_index(drop=True)
        # REVIEW is excluded for formal supervision, not repaired by fractional weighting.
        self.windows=self.windows[~self.windows.review & self.windows.complete_state_sequence].reset_index(drop=True)
        from .pahbd_phase import prepare_phase,annotate_hall
        reference_contract=reference_contract or Path(DEFAULT_REGISTRY).parents[1]/'pahbd/phase_reference.json'
        self.phase_contract=json.loads(Path(reference_contract).read_text())
        if self.phase_contract['schema']!='pahbd_hall_phase_v2':raise ValueError('Hall phase contract required')
        phase=pd.read_parquet(self.root/f'native_wing_phase_{split}.parquet')
        self.phase_tables={fid:prepare_phase(t,self.provenance[fid]['transmission_ratio'],self.phase_contract)
                           for fid,t in phase.groupby('flight_id') if fid in self.provenance}
        samples=pd.read_parquet(self.root/f'samples_{split}.parquet',columns=['flight_id','timestamp_us'])
        starts={fid:t.timestamp_us.to_numpy() for fid,t in samples.groupby('flight_id',sort=False)}
        self.windows=annotate_hall(self.windows,self.provenance,self.phase_tables,starts,self.phase_contract)
        self.windows['phase_policy']=phase_policy
        self._cached_id=None;self.tables={}

    def tensor(self,value):return torch.as_tensor(value,dtype=self.dtype,device=self.device)
    def __len__(self):return len(self.windows)

    def _load(self,fid):
        if self._cached_id==fid:return
        self.tables={name:pd.read_parquet(self.root/f'native_{name}_{self.split}.parquet',
                      filters=[('flight_id','==',fid)]).sort_values('timestamp_us') for name in TOPICS if name!='wing_phase'}
        self.tables['wing_phase']=self.phase_tables[fid]
        self.samples=pd.read_parquet(self.root/f'samples_{self.split}.parquet',filters=[('flight_id','==',fid)])
        self._cached_id=fid

    def _observation(self,name,row,decision,meta):
        measurement=int(row.timestamp_sample_us) if int(row.timestamp_sample_us)>0 else int(row.timestamp_us)
        if name in ('wing_phase','flap_frequency'):measurement=int(row.timestamp_us)
        availability=int(row.get('availability_us',row.timestamp_us))
        value=np.zeros(15);value[6]=1;mask=np.zeros(14);variance=np.ones(14);quality=np.ones(14)
        if name=='vehicle_local_position':
            value[:6]=row[FIELDS[name]].to_numpy(float)
            mask[:6]=[row.xy_valid,row.xy_valid,row.z_valid,row.v_xy_valid,row.v_xy_valid,row.v_z_valid]
            std=np.array([row.eph,row.eph,row.epv,row.evh,row.evh,row.evv],float)
            variance[:6]=np.clip(np.nan_to_num(std,nan=10.,posinf=10.),.05,10.)**2
            if row.dead_reckoning:quality[:6]*=.25
        elif name=='vehicle_attitude':
            value[6:10]=row[FIELDS[name]].to_numpy(float);mask[6:9]=1;variance[6:9]=.03**2
            norm=np.linalg.norm(value[6:10])
            if not np.isfinite(norm) or norm<.5:mask[6:9]=0;value[6:10]=[1,0,0,0]
            else:value[6:10]/=norm
        elif name=='vehicle_angular_velocity':
            value[10:13]=row[FIELDS[name]].to_numpy(float);mask[9:12]=1;variance[9:12]=.05**2
        elif name=='flap_frequency':
            value[13]=row.frequency_hz;mask[12]=float(np.isfinite(value[13]) and value[13]>0);variance[12]=.2**2
        elif name=='wing_phase':
            value[14]=row.phase_rad
            mask[13]=float(row.phase_usable);variance[13]=self.phase_contract['phase_variance_rad2']
        else:raise ValueError(name)
        finite=np.r_[np.isfinite(value[:6]),np.repeat(np.isfinite(value[6:10]).all(),3),np.isfinite(value[10:])]
        mask*=finite
        for a,b in meta['review_intervals_us']:
            if a<=availability<=b or a<=measurement<=b:quality*=.25
        value=np.nan_to_num(value,nan=0.,posinf=0.,neginf=0.)
        return Observation((measurement-decision)*1e-6,(availability-decision)*1e-6,
                           self.tensor(value),self.tensor(mask),self.tensor(variance),self.tensor(quality),name)

    def __getitem__(self,index):
        w=self.windows.iloc[index];fid=w.flight_id;self._load(fid);meta=self.provenance[fid]
        shared=bool(w.shared_periodic_eligible)
        parameters=meta['parameters']
        if any(parameters.get(k)!=v for k,v in self.contract['required_mapping_parameters'].items()):
            raise ValueError('unverified actuator channel mapping')
        segment=self.samples.iloc[int(w.start_sample):int(w.end_sample)+1]
        if not (segment.segment_id==w.segment_id).all():raise ValueError('window crosses segment boundary')
        decision=int(w.decision_us);start_us=int(segment.iloc[0].timestamp_us)
        lower=int(w.support_start_us);end=int(w.last_target_measurement_us)
        if not meta['start_us']<=lower<=start_us<decision<end<=meta['end_us']:
            raise ValueError('window outside flight support')
        start=(start_us-decision)*1e-6
        events=[];targets=[];seeds={}
        for name in TOPICS[:5]:
            table=self.tables[name]
            mt=table.timestamp_sample_us.where(table.timestamp_sample_us>0,table.timestamp_us)
            if name in ('wing_phase','flap_frequency'):mt=table.timestamp_us
            av=table.get('availability_us',table.timestamp_us)
            # Future observations retained ONLY as separate loss targets for rigid-body state.
            selected=table[(mt>=lower)&(mt<=end)&(av>=mt)&table.in_reviewed_flight]
            for _,row in selected.iterrows():
                e=self._observation(name,row,decision,meta)
                if e.availability<=start:
                    seeds[name]=e
                elif e.availability<=0:events.append(e)
                if name in FIELDS and 0<e.measurement<=(end-decision)*1e-6:
                    targets.append(e)
        for name in FIELDS:
            if name not in seeds or not bool(seeds[name].mask[{'vehicle_local_position':slice(0,6),'vehicle_attitude':slice(6,9),'vehicle_angular_velocity':slice(9,12)}[name]].all()):raise ValueError('missing causal bootstrap state')
        group=self.contract['groups'][w.sampling_group]
        limits={'vehicle_local_position':group['maximum_state_gap_us']*1e-6,
                'vehicle_attitude':group['history_attitude_age_us']*1e-6,
                'vehicle_angular_velocity':self.contract['history_rate_age_us']*1e-6}
        for name in FIELDS:
            if start-seeds[name].measurement>limits[name]:raise ValueError('stale bootstrap state')
        x=torch.cat((seeds['vehicle_local_position'].value[:6],seeds['vehicle_attitude'].value[6:10],
                     seeds['vehicle_angular_velocity'].value[10:13]))
        # Origin translation only, inferred from history; no target-derived normalization.
        origin=x[:3].clone();x=x.clone();x[:3]-=origin
        def centered(e):
            v=e.value.clone();v[:3]-=origin
            return replace(e,value=v)
        events=[centered(e) for e in events];targets=[centered(e) for e in targets]
        controls=self._controls(start_us,end,decision,lower)
        fseed=seeds.get('flap_frequency')
        valid=(fseed is not None and bool(fseed.mask[12]) and start-fseed.measurement<=.12
               and bool(w.shared_periodic_eligible))
        periodic=valid and self.phase_policy!='average'
        if fseed is not None and bool(fseed.mask[12]):f=fseed.value[13:14].clone()
        else:
            b0,bu,bf=self.phase_coefficients
            f=-(b0+bu*controls.values[0,:1])/bf
        # No phase origin prior: first reliable Hall event initializes the circular
        # state at measurement time in the causal replay, then flow propagates it.
        phi=self.tensor([0.])
        if not periodic:
            events=[replace(e,mask=torch.cat((e.mask[:13],torch.zeros_like(e.mask[13:])))) for e in events]
        covariance=torch.diag(self.tensor([.04,.2]))
        initial=State(x,self.tensor(np.zeros(12)),f,phi,controls.values[0,1:].clone(),
                      torch.cat((self.tensor([1.]*6+[.1]*3+[1.]*3),covariance.diagonal())),
                      self.tensor([0.]),phase_covariance=covariance)
        metadata=dict(window_id=w.window_id,flight_id=fid,split=self.split,segment_id=int(w.segment_id),
                      decision_us=decision,support_start_us=lower,support_end_us=int(w.support_end_us),
                      manifest_sha256=self.manifest_sha256,phase_valid=bool(valid),
                      phase_reference=('hall_circular' if shared else 'periodic_disabled'),
                      phase_policy=self.phase_policy,phase_mode=w.phase_mode,periodic_enabled=bool(periodic),
                      phase_reference_id=w.phase_reference_id,
                      shared_periodic_eligible=bool(w.shared_periodic_eligible),
                      phase_contract=self.phase_contract,phase_exclusion=w.phase_exclusion,
                      horizon_us=int(w.horizon_us),actual_horizon_s=(end-decision)*1e-6,
                      phase_initialization='Hall circular anchor + autonomous ZOH PhasePropagator',
                      bootstrap_measurements={k:v.measurement for k,v in seeds.items()},
                      controls='logged_ZOH_conditional_replay',control_mapping=self.contract['control_mapping'],
                      sampling_group=w.sampling_group,review=bool(w.review))
        targets.sort(key=lambda e:e.measurement)
        # OFFLINE target-only encoder reference: never passed to encode/flow.
        enc=pd.read_parquet(self.root/f'native_encoder_count_{self.split}.parquet',filters=[('flight_id','==',fid)]).sort_values('timestamp_us')
        enc=enc[(enc.timestamp_us>=lower)&(enc.timestamp_us<=int(w.support_end_us))]
        et=enc.timestamp_us.to_numpy();ep=(enc.total_count.to_numpy()-meta['phase_origin_count'])*2*math.pi/(4096*meta['transmission_ratio'])
        phases=[]
        for e in targets:
            t=decision+round(e.measurement*1e6);j=np.searchsorted(et,t)
            if j==0 or j>=len(et) or et[j]-et[j-1]>30000:phases.append(None)
            else:phases.append(float(np.interp(t,et[j-1:j+1],ep[j-1:j+1])))
        pairs=[]
        if valid:
            for i,e in enumerate(targets):
                if phases[i] is None:continue
                candidates=[(j,abs(phases[j]-phases[i]-2*math.pi)) for j in range(i+1,len(targets))
                            if targets[j].source==e.source and phases[j] is not None]
                if candidates:
                    j,error=min(candidates,key=lambda pair:pair[1])
                    neighbors=[abs(phases[k]-phases[j]) for k in range(len(targets))
                               if k!=j and targets[k].source==e.source and phases[k] is not None]
                    tolerance=min(neighbors)/2 if neighbors else 0.
                    if error<=tolerance:pairs.append((i,j))
        metadata.update(cycle_pairs=pairs,target_phase_reference=phases,
                        cycle_section_tolerance_rad='half nearest native same-source phase spacing',
                        cycle_reference='offline_encoder_bracket_30ms_native_state_pairs_not_exact_wing_sections')
        return Window(initial,start,sorted(events,key=lambda e:e.availability),controls,targets,metadata)

    def _controls(self,start,end,decision,lower):
        tabs=[self.tables[n] for n in ('actuator_motors','actuator_servos')]
        ts=sorted(set([start,end]+[int(t) for tab in tabs for t in tab.timestamp_us if start<t<=end]))
        out=[]
        for t in ts:
            vals=[]
            for tab,fields in zip(tabs,[['control[0]'],[f'control[{i}]' for i in range(3)]]):
                j=np.searchsorted(tab.timestamp_us,t,side='right')-1
                if j<0:raise ValueError('missing ZOH control')
                row=tab.iloc[j]
                if row.timestamp_us<lower or t-row.timestamp_us>self.contract['control_age_us']:
                    raise ValueError('stale/out-of-segment control')
                vals.extend(row[fields].to_numpy(float))
            out.append(vals)
        return ControlTape([(t-decision)*1e-6 for t in ts],self.tensor(np.asarray(out))).canonical()


def configure_latent(window,config):
    """Data has no trainable window initial states; h always starts at zero."""
    window.initial=window.initial.with_(h=window.initial.x.new_zeros(config.latent))
    return window


class ConfiguredWindows:
    """Lazy ragged-window sequence; keeps only the current flight's native tables cached."""
    def __init__(self,dataset,indices,config):
        self.dataset=dataset;self.indices=list(indices);self.config=config

    def __len__(self):return len(self.indices)

    @property
    def index(self):
        return self.dataset.windows.iloc[self.indices].reset_index(drop=True)

    def __getitem__(self,index):
        return configure_latent(self.dataset[self.indices[index]],self.config)
