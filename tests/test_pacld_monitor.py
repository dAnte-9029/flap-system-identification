import subprocess
from system_identification.training.pacld_monitor import Monitor,read_gpu


def test_eta_and_resumed_progress(tmp_path):
    s=dict(optimizer_steps=64,windows_seen=2048,best_score=25.,best_step=64)
    m=Monitor(tmp_path/'progress.log',17,0,32,342,32,s)
    for _ in range(25):m.observe(dict(end_to_end_seconds=3.),{})
    m.status('TRAINING')
    assert 'pending calibration' in (tmp_path/'progress.log').read_text()
    m.validation_end(64,25.,30.,True);s['monitor']=m.snapshot();m.close()
    resumed=Monitor(tmp_path/'resumed.log',17,0,32,342,32,s);resumed.observe(dict(end_to_end_seconds=3.),{});resumed.status('TRAINING');resumed.close()
    text=(tmp_path/'resumed.log').read_text()
    assert '2048 / 10944' in text and 'ETA CALIBRATING' not in text
    assert '00:18:24' in text # 278*3 seconds + 9*30 seconds


def test_telemetry_failure_is_na(monkeypatch):
    def fail(*a,**kw):raise subprocess.TimeoutExpired('nvidia-smi',2)
    monkeypatch.setattr(subprocess,'check_output',fail)
    assert read_gpu(0)==dict(utilization_percent=None,memory_gib=None)


def test_gpu_config_only_changes_monitoring_cadence():
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    cpu=json.loads((root/'configs/pacld/default.json').read_text())
    gpu=json.loads((root/'configs/pacld/gpu.json').read_text())
    assert cpu['training']['validation_interval']==1000
    assert gpu['training']['validation_interval']==32
    assert gpu['training']['terminal_log_interval']==10
    gpu['training']['validation_interval']=cpu['training']['validation_interval']
    gpu['training']['terminal_log_interval']=cpu['training']['terminal_log_interval']
    assert gpu==cpu


def test_pending_final_validation_is_in_eta(tmp_path):
    s=dict(optimizer_steps=33,windows_seen=1056,best_score=25.)
    m=Monitor(tmp_path/'progress.log',17,0,32,33,32,s)
    for _ in range(25):m.observe(dict(end_to_end_seconds=3.),{})
    m.validation_end(32,25.,30.,True)
    m.status('VALIDATING');m.close()
    assert 'ETA remaining : 00:00:30' in (tmp_path/'progress.log').read_text()
