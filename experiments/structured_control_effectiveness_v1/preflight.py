"""Check exact old caches/metadata without rebuilding or touching test artifacts."""
from common import *
import torch, yaml, shutil
from model import make_model
def main():
    cfg=json.loads((HERE/'config.json').read_text());manifest=json.loads((DATA/'dataset_manifest.json').read_text())
    registry=yaml.safe_load((ROOT/cfg['registry']).read_text())['datasets'][cfg['dataset_id']]
    assert sha(DATA/'dataset_manifest.json')==registry['manifest_sha256']
    audit=json.loads((TRAIN/'data_audit.json').read_text());archive=json.loads((TRAIN/'archive_manifest.json').read_text())
    records={};meta={}
    for split in ['train','validation']:
        file=CACHE/f'{split}.npz';expected=audit['partitions'][split]['cache_sha256'];assert sha(file)==expected;records[str(file)]=expected
        cached=np.load(file);mask=cached['history_mask'];length=cached['history_length']
        assert mask[np.arange(len(length)),length-1,13:17].all(), 'reference commands must be valid at the causal history endpoint'
        for name in [f'{split}_windows.parquet','normalization.json','train_1000ms/config.json']:
            assert sha(TRAIN/name)==archive['files'][name]['sha256'];records[str(TRAIN/name)]=sha(TRAIN/name)
        for name in [f'windows_{split}.parquet',f'samples_{split}.parquet']:
            assert sha(DATA/name)==manifest['artifact_sha256'][name];records[str(DATA/name)]=manifest['artifact_sha256'][name]
        meta[split]=pd.read_parquet(TRAIN/f'{split}_windows.parquet')
        assert len(meta[split])==cfg['expected_windows'][split]
        assert meta[split].flight_id.nunique()==cfg['expected_flights'][split]
        registered=pd.read_parquet(DATA/f'windows_{split}.parquet');registered=registered[(registered.sampling_group=='main_50hz')&(registered.history_us==500000)&(registered.horizon_us==1000000)]
        fields=['flight_id','origin_sample','start_sample','end_sample','history_count','future_count']
        pd.testing.assert_frame_equal(meta[split][fields].reset_index(drop=True),registered[fields].reset_index(drop=True),check_dtype=False)
    assert set(meta['train'].flight_id).isdisjoint(meta['validation'].flight_id)
    norm=json.loads((TRAIN/'normalization.json').read_text());shutil.copyfile(TRAIN/'normalization.json',HERE/'normalization.json')
    for name in ['pairs','cases','windows','cohorts']:
        shutil.copyfile(SWAP/'results'/f'{name}.csv',OUT/f'{name}.csv')
    counts={name:sum(p.numel() for p in make_model(name,norm).parameters()) for name in cfg['models']}
    assert all(abs(n/counts['M0']-1)<.25 for n in counts.values())
    for folder in [SWAP,ABLATION]:
        for entry in json.loads((folder/'archive_manifest.json').read_text())['files']:
            p=folder/entry['path'];assert sha(p)==entry['sha256'];records[str(p)]=entry['sha256']
    for p in [ROOT/'src/system_identification/models/velocity_rate_gru.py',ROOT/'src/system_identification/models/control_events.py',ROOT/'scripts/train_velocity_rate_horizons.py',ROOT/'scripts/train_velocity_rate_learning.py',ROOT/'scripts/prepare_velocity_rate_learning.py']:
        records[str(p)]=sha(p)
    save(OUT/'source_receipt.json',dict(base_sha='2c99a7fcf8495cfb1d21535e928716537aa7eb73',dataset_id=cfg['dataset_id'],manifest_sha256=sha(DATA/'dataset_manifest.json'),files=records,parameters=counts,train_windows=len(meta['train']),validation_windows=len(meta['validation']),train_flights=94,validation_flights=27,test_read=False,repacked=False,normalization_refitted=False,config_sha256=sha(HERE/'config.json')))
    print('Verified frozen train/validation caches, window identities, upstream artifacts; parameters',counts,flush=True)
if __name__=='__main__':main()
