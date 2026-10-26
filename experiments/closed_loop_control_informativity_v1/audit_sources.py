"""Recover exact firmware source paths and preserve bounded evidence; no PX4 writes."""
from extract import *
import requests,concurrent.futures,platform,subprocess
PATHS=[
'src/modules/fw_mode_manager/FixedWingModeManager.cpp',
'src/modules/fw_att_control/FixedwingAttitudeControl.cpp',
'src/modules/fw_rate_control/FixedwingRateControl.cpp',
'src/lib/rate_control/rate_control.cpp',
'src/modules/control_allocator/ControlAllocator.cpp',
'src/lib/control_allocation/control_allocation/ControlAllocationPseudoInverse.cpp',
'src/modules/navigator/mission.cpp',
]
def fetch(item):
 sha,path=item;url=f'https://raw.githubusercontent.com/jiashunjiang2022/PX4-Autopilot/{sha}/{path}'
 try:
  r=requests.get(url,timeout=25);r.raise_for_status();b=r.content;h=hashlib.sha256(b).hexdigest();dest=HERE/'sources'/f'{h}_{Path(path).name}'
  if not dest.exists():dest.write_bytes(b)
  return dict(commit=sha,path=path,url=url,status='exact_commit_source',sha256=h,local=str(dest.relative_to(HERE)))
 except Exception as e:return dict(commit=sha,path=path,url=url,status='unavailable',reason=str(e))
def main():
 meta=pd.read_csv(RES/'windows.csv.gz');shas=sorted(meta.firmware.dropna().unique())
 jobs=[(s,p) for s in shas for p in PATHS]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:out=list(ex.map(fetch,jobs))
 pd.DataFrame(out).to_csv(RES/'firmware_sources.csv',index=False)
 extra=[(c,p) for c in shas for p in ['src/modules/navigator/mission_base.cpp','src/modules/navigator/mission_block.cpp']]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:nav=list(ex.map(fetch,extra))
 pd.DataFrame(nav).to_csv(RES/'navigator_sources.csv',index=False)
 terms=['_pos_sp_triplet','_rates_sp','rate_error','_rate_control.update','_vehicle_torque_setpoint.timestamp','normalizeControlAllocationMatrix','clipActuatorSetpoint','_mission_result.seq_current','is_mission_item_reached']
 evidence=[]
 for r in out:
  if r['status']!='exact_commit_source':continue
  lines=(HERE/r['local']).read_text().splitlines()
  for i,line in enumerate(lines):
   if any(s in line for s in terms):evidence.append(dict(commit=r['commit'],path=r['path'],line=i+1,text=line.strip(),sha256=r['sha256']))
 pd.DataFrame(evidence).to_csv(RES/'source_evidence.csv',index=False)
 # Prior source hashes and frozen selection provenance are checked without mutating originals.
 files=[SCOPE/'config.json',SCOPE/'results/frozen_selection.json']
 for sp in CFG['partitions']:files.extend([SCOPE/f'results/{sp}/candidate_index.csv.gz',SCOPE/f'results/{sp}/command_events.csv.gz',SCOPE/f'results/{sp}/controller_headers.json'])
 for exp,names in {'local_control_response_rollout_v1':['results/pairs.csv','results/input_causality_check.json','results/rate_clock_audit.csv'],'structured_control_effectiveness_group_holdout_v1':['results/factual_points.csv.gz','results/cohorts.csv','results/component_assignments.csv','report_zh.md'],'local_control_response_time_contract_ablation_v1':['time_contract_zh.md','sources/stage4_report.txt','results/firmware_sources.csv']}.items():
  files.extend(HERE.parent/exp/n for n in names)
 write_json(RES/'upstream_hashes.json',{str(p.relative_to(ROOT)):old.sha(p) for p in files})
 write_json(RES/'environment.json',dict(python=sys.executable,python_version=sys.version,numpy=np.__version__,pandas=pd.__version__,platform=platform.platform(),base_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),gpu_used=False))
 # Publisher final version separate from misleadingly named old Drive preprint.
 url='https://pure.tudelft.nl/ws/portalfiles/portal/257289673/1-s2.0-S0005109825004510-main.pdf'
 try:
  r=requests.get(url,timeout=30);r.raise_for_status();assert r.content.startswith(b'%PDF')
  p=HERE/'sources/Dinkla_published.pdf';p.write_bytes(r.content)
  subprocess.run(['pdftotext',str(p),str(HERE/'sources/Dinkla_published.txt')],check=True)
  write_json(HERE/'sources/dinkla_published_receipt.json',dict(url=url,sha256=old.sha(p),status='downloaded_final'))
 except Exception as e:write_json(HERE/'sources/dinkla_published_receipt.json',dict(url=url,status='unavailable',reason=str(e)))
if __name__=='__main__':main()
