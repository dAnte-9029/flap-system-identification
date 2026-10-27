"""Small exact-identity spot check, using archived B nodes rather than refitting/interpolation."""
from common import *

def main():
 frames=[pd.read_parquet(CAUSAL/'results/history_states.parquet',columns=['split','flight_id','sample_index','B_p','B_q','B_r']),pd.read_csv(FACTOR/'results/restored_B_tail.csv.gz')];table=pd.concat(frames).set_index(['split','flight_id','sample_index']);rows=[]
 for split in ['train','validation']:
  pack={k:v for k,v in target_pack(split).items()};meta=pd.read_parquet(TRAIN/f'{split}_windows.parquet');ids=np.random.default_rng(17).choice(len(meta),min(1024,len(meta)),replace=False)
  for idx in ids:
   m=meta.iloc[idx];n=int(pack['loss_mask'][idx].sum())
   for k in sorted(set([0,n//2,n-1])):
    expected=table.loc[(split,m.flight_id,int(m.origin_sample+k+1)),['B_p','B_q','B_r']].to_numpy(dtype=np.float32);np.testing.assert_array_equal(expected,pack['rate_B'][idx,k])
  rows.append(dict(split=split,windows=len(ids),nodes=3*len(ids),exact_archived_B_float32=True))
 save(OUT/'packing_verification.json',dict(passed=True,checks=rows,source='previous archived B by exact flight/sample identity; no nearest time join'))
 print('Archived B packing exact in sampled first/middle/final supported nodes')
if __name__=='__main__':main()
