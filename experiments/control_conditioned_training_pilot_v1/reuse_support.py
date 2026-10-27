"""Reuse unchanged fold-input support audit; weights do not change marginal ranges."""
from common import *
import shutil

def main():
 archive=json.loads((HOLDOUT/'archive_manifest.json').read_text());lookup={r['path']:r['sha256'] for r in archive['files']};receipt=json.loads((HOLDOUT/'results/source_receipt.json').read_text());oldhash=[v for k,v in receipt['files'].items() if k.endswith('local_control_response_swap_sensitivity_v1/results/inputs.npz')];assert oldhash==[sha(SWAP/'results/inputs.npz')]
 ref=json.loads((HOLDOUT/'results/fold_support_reference.json').read_text());indices=np.load(OUT/'holdout_window_indices.npz')
 for f in ref['folds']:assert f['train_indices_sha256']==hashlib.sha256(indices[f"fold{f['fold']}_train_indices"].tobytes()).hexdigest()
 files={}
 for name in ['fold_control_support.csv','fold_support_reference.json','excluded_validation_background.csv']:
  src=HOLDOUT/'results'/name;assert sha(src)==lookup['results/'+name];shutil.copyfile(src,OUT/name);files[str(src)]=sha(src)
 x=np.load(SWAP/'results/inputs.npz');cases=pd.read_csv(OUT/'cases.csv');table=pd.read_csv(OUT/'fold_control_support.csv')
 for i in table.case_id.unique():
  other=(i//4)*4+(0 if cases.cell.iloc[i] in ['AA','AB'] else 2)
  for key in ['history','history_mask','history_dt','history_length','initial','dt']:np.testing.assert_array_equal(x[key][i],x[key][other])
 save(OUT/'support_reuse_receipt.json',dict(passed=True,files=files,inputs_sha256=sha(SWAP/'results/inputs.npz'),same_fit_members=True,receiver_noncontrol_inputs_exact=True,reference='unchanged fold-fit marginal extrema, not joint/counterfactual support',test_read=False))
 text=table.groupby('fold')[['control_outside_fold_marginal','initial_outside_fold_marginal','physical_range_violation']].sum().to_string()
 print('\n'.join(line.rstrip() for line in text.splitlines()))
if __name__=='__main__':main()
