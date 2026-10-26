"""Read exact flight firmware Git objects; no current PX4 files, ULogs or test data."""
from common import *
import subprocess,shutil,urllib.request
from concurrent.futures import ThreadPoolExecutor
PATHS=['src/drivers/encoder/as5600/AS5600.cpp','src/drivers/encoder/as5600/AS5600Math.hpp','src/drivers/rpm_capture/RPMCapture.cpp','src/modules/wing_phase/WingPhase.cpp','src/modules/wing_phase/WingPhaseMath.cpp','src/modules/sensors/vehicle_angular_velocity/VehicleAngularVelocity.cpp','src/modules/ekf2/EKF2.cpp','src/modules/ekf2/EKF/estimator_interface.h','src/modules/ekf2/EKF/estimator_interface.cpp','src/modules/ekf2/EKF/output_predictor/output_predictor.h','msg/EncoderCount.msg','msg/FlapFrequency.msg','msg/HallEvent.msg','msg/WingPhase.msg','msg/versioned/VehicleAngularVelocity.msg','msg/versioned/VehicleLocalPosition.msg']
PATTERNS={'AS5600.cpp':['read_raw_angle(angle_raw)','const hrt_abstime now','alpha =','flap_frequency.timestamp','enc.timestamp','publish(enc)','ScheduleOnInterval'], 'AS5600Math.hpp':['flap_frequency_from_rpm','fabsf','ratio'], 'RPMCapture.cpp':['hall.timestamp','gpio_interrupt_callback','hrt_absolute_time','_hrt_timestamp ='], 'WingPhase.cpp':['interpolate_count_at_timestamp','compute_phase','message.timestamp','_hall_sub.update','_encoder_sub.update'], 'WingPhaseMath.cpp':['counts_per_cycle','phase_rad'], 'VehicleAngularVelocity.cpp':['CalibrateAndPublish','timestamp_sample =','timestamp = hrt_absolute_time','FilterAngularVelocity'], 'EKF2.cpp':['PublishLocalPosition(now)','lpos.timestamp_sample','lpos.timestamp =','const hrt_abstime now','imu_sample.time_us','getPosition','getVelocity']}
def main():
 OUT.mkdir(parents=True,exist_ok=True);dest=HERE/'sources' if OUT==HERE/'results' else OUT.parent/'sources';dest.mkdir(exist_ok=True);w=pd.read_csv(SWAP/'results/windows.csv');records=[];lines=[]
 snapshot_manifest=HERE/'results/firmware_sources.csv'
 previous={(r.commit,r.path):r for r in pd.read_csv(snapshot_manifest).itertuples()} if snapshot_manifest.exists() else {}
 cache=DATA/'firmware_sources';cached={(r['commit'],r['source']):r for r in json.loads((cache/'source_manifest.json').read_text())}
 def read(item):
  commit,path=item;g=w[w.firmware==commit];p=subprocess.run(['git','-C',str(PX4),'show',f'{commit}:{path}'],capture_output=True);origin='local_git_object';data=p.stdout if p.returncode==0 else None
  key=(commit,path);cp=cache/commit/path
  if data is None and key in cached and cp.exists():
   assert sha(cp)==cached[key]['sha256'];data=cp.read_bytes();origin='registered_dataset_source_cache'
  if data is None and key in previous and previous[key].status=='available_exact_commit':
   previous_path=HERE/previous[key].snapshot;assert sha(previous_path)==previous[key].sha256;data=previous_path.read_bytes();origin='previous_exact_commit_snapshot'
  if data is None:
   try:
    with urllib.request.urlopen(f'https://raw.githubusercontent.com/jiashunjiang2022/PX4-Autopilot/{commit}/{path}',timeout=25) as response:data=response.read()
    origin='exact_commit_https'
   except Exception as e:return dict(commit=commit,path=path,status='unavailable',error=str(e),windows=len(g)),[]
  h=hashlib.sha256(data).hexdigest();name=h+'_'+Path(path).name;(dest/name).write_bytes(data)
  blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
  if origin=='local_git_object':assert blob==subprocess.check_output(['git','-C',str(PX4),'rev-parse',f'{commit}:{path}'],text=True).strip()
  rec=dict(commit=commit,path=path,status='available_exact_commit',origin=origin,windows=len(g),flights=g.flight_id.nunique(),sha256=h,git_blob_computed=blob,snapshot='sources/'+name)
  patterns=PATTERNS.get(Path(path).name,['timestamp','frequency','phase','count','getPosition','getVelocity','_output_predictor','_output_new.pos']);ls=[dict(commit=commit,path=path,line=i,text=line.strip(),sha256=h) for i,line in enumerate(data.decode().splitlines(),1) if any(s in line for s in patterns)]
  return rec,ls
 with ThreadPoolExecutor(max_workers=8) as pool:
  for rec,ls in pool.map(read,[(c,p) for c in sorted(w.firmware.unique()) for p in PATHS]):records.append(rec);lines.extend(ls)
 pd.DataFrame(records).to_csv(OUT/'firmware_sources.csv',index=False);pd.DataFrame(lines).to_csv(OUT/'source_evidence.csv',index=False)
 oldpath=Path('/home/zn/flap-system-identification/.worktrees/local-control-response-scope-v1/experiments/local_control_response_scope_v1/sources/stage4_report.txt')
 if not oldpath.exists():
  receipt=json.loads((HERE/'results/source_receipt.json').read_text());oldpath=HERE/'sources/stage4_report.txt';assert sha(oldpath)==receipt['stage4_sha256']
 if oldpath.resolve()!=(dest/'stage4_report.txt').resolve():shutil.copyfile(oldpath,dest/'stage4_report.txt')
 used=['src/system_identification/data/velocity_rate_dataset.py','src/system_identification/data/trajectory_dataset.py','src/system_identification/models/velocity_rate_gru.py','src/system_identification/models/control_events.py','experiments/control_response_eval_v1/evaluate.py']
 save(OUT/'source_receipt.json',dict(base_sha='ae8598a94555c908500856cc415cf2119c396078',firmware_commits=sorted(w.firmware.unique()),PX4_read_only=True,stage4_source_path=str(oldpath),stage4_sha256=sha(oldpath),python_sources_sha256={p:sha(ROOT/p) for p in used},test_read=False))
 print('Exact firmware snapshots:',sum(r['status']=='available_exact_commit' for r in records),'commits:',w.firmware.nunique(),'distinct snapshots:',len(list(dest.glob('*'))))
if __name__=='__main__':main()
