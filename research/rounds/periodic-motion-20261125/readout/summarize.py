"""Summarize shared-backbone comparisons independently of end-to-end optimization."""
from pathlib import Path
import json
import runpy
import numpy as np

HERE=Path(__file__).resolve().parent
aggregate=runpy.run_path(str(HERE.parent/'analyze_accuracy.py'))['aggregate']
read=lambda p:json.loads(p.read_text())


def main():
    seeds=['17','29','43']
    pairs=(('strong','corrected'),('matched','corrected'),('held','corrected'),
           ('strong','matched'),('strong','held'),('matched','held'))
    table={s:read(HERE/'runs'/s/'flight_mse.json') for s in seeds+['incumbent17']}
    main=aggregate({s:table[s] for s in seeds},pairs)
    incumbent=aggregate({'17':table['incumbent17']},pairs)
    original=read(HERE.parents[1]/'history-identification-20261125/extended/runs/low/final/metrics.json')
    for k,o in original.items():
        assert abs(incumbent['metrics']['strong'][k]['mean']-o)<2e-6
    direct={s:read(HERE.parent/'clock'/f'seed{s}_paired.json')['flight_mse'] for s in seeds}
    frozen=aggregate(direct,[('held','corrected')])
    base=read(HERE.parent/'accuracy.json')['results']['common']
    for s in seeds:
        for k in ('v05','w05','v1','w1','v2','w2'):
            difference=main['metrics']['strong'][k]['seeds'][s]-base['metrics']['strong'][k]['seeds'][s]
            assert abs(difference)<2e-6,(s,k,difference)
            assert abs(frozen['metrics']['held'][k]['seeds'][s]-base['metrics']['periodic'][k]['seeds'][s])<2e-6
    canonical=aggregate({s:read(HERE/'canonical_runs'/s/'flight_mse.json') for s in seeds},pairs)
    incumbent_canonical=aggregate({'17':read(HERE/'canonical_runs/incumbent17/flight_mse.json')},pairs)
    practical=aggregate({s:read(HERE/'original_runs'/('incumbent'+s)/'flight_mse.json') for s in seeds},pairs)
    for s in seeds:
        origin=(HERE.parents[1]/'history-identification-20261125/extended/runs/low/step49152' if s=='17'
                else HERE/'backbones'/('incumbent'+s)/'step49152')
        reference=read(origin/'metrics.json')
        for name,v in reference.items():
            assert abs(practical['metrics']['strong'][name]['seeds'][s]-v)<2e-6
    final_rows={}
    terminal_steps={}
    for s in seeds:
        stop=(HERE.parents[1]/'history-identification-20261125/extended/runs/low/stop.json' if s=='17'
              else HERE/'backbones'/('incumbent'+s)/'stop.json')
        decision=read(stop)
        assert decision['reason']=='validation_plateau' and decision['last_16_at_min_lr']
        terminal_steps[s]=decision['step']
        row=('incumbent'+s if decision['step']==49152 else 'terminal'+s)
        final_rows[s]=read(HERE/'original_runs'/row/'flight_mse.json')
    practical_final=aggregate(final_rows,pairs)
    (HERE/'results.json').write_text(json.dumps(dict(matched_fits=main,incumbent=incumbent,replace_phase_only=frozen,
        canonical_path=canonical,incumbent_canonical=incumbent_canonical,original_protocol=practical,original_terminal=practical_final,original_terminal_steps=terminal_steps),indent=2)+'\n')


if __name__=='__main__':main()
