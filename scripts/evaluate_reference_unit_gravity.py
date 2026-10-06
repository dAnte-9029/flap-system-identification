"""Run nominal, complete-context and bounded-sensitivity checks with explicit units."""
import shutil
import argparse
import torch
import train_reference_unit_gravity as fixed
import evaluate_control_windows_v3 as nominal
import control_windows_v3_response as response
import control_windows_v3_action_probe as probe
import diagnose_reference_unit_gravity as diagnosis


def main(stage='all'):
    torch.set_num_threads(1)
    nominal.OUT=fixed.OUT;nominal.ART=fixed.ART
    if stage in ['all','nominal']:nominal.main(expected_models=15,baseline_folder=fixed.ART)
    response.OUT=fixed.OUT;response.ART=fixed.ART;response.model_from=fixed.model_from
    if stage in ['all','response']:response.main(baseline_folder=fixed.ART)
    # Ridge uses exactly the same raw unit-direction inputs in both stages.
    probe.OUT=fixed.OUT;probe.ART=fixed.ART;probe.model_from=fixed.model_from
    if stage in ['all','probe']:
        shutil.copytree(fixed.PARENT/'ridge',fixed.OUT/'ridge')
        probe.main()
    if stage in ['all','diagnosis']:diagnosis.main()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',default='all',choices=['all','nominal','response','probe','diagnosis'])
    main(parser.parse_args().stage)
