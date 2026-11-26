"""Flight contract and cached training-only projection onto DLinear's regular output grid."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ROOT = Path("/home/zn/flap-system-identification")
HERE = ROOT / ".worktrees/psloss-flight-transfer"
FIELDS = ROOT / ".worktrees/mdpnet-flight-transfer/outputs/mdpnet_flight_fields"
HIST = ROOT / "outputs/history_slow_fast_20261119_cache"
TAPE = ROOT / "outputs/controlled_slow_fast_20261119_tapes"
OUTPUT = HERE / "outputs/psloss_flight"
CACHE = OUTPUT / "data"
DT = .02
H = 102
STRUCT_STEPS = 95  # Up to 1.90 s, fully bracketed even in the slower sampling group.


def write(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def integral(tape, queries):
    """Exact prefix integral of a piecewise-constant four-channel command; units command*seconds."""
    u = tape[..., :4].reshape(len(tape), -1, 4).astype("float64")
    dt = tape[..., 4].reshape(len(tape), -1).astype("float64") * 1e-6
    ends = dt.cumsum(1)
    starts = ends - dt
    before = np.concatenate([np.zeros((len(tape),1,4)), (u*dt[...,None]).cumsum(1)[:,:-1]],axis=1)
    q = queries.reshape(len(tape),-1).astype("float64")
    base = np.arange(len(tape))[:,None]*10.
    idx = np.searchsorted((base+ends).ravel(),(base+q).ravel(),"left").reshape(q.shape)
    idx = np.clip(idx - np.arange(len(q))[:,None]*ends.shape[1], 0, ends.shape[1]-1)
    rows = np.arange(len(q))[:,None]
    elapsed = np.clip(q-starts[rows,idx],0,dt[rows,idx])
    return (before[rows,idx]+elapsed[...,None]*u[rows,idx]).reshape(*queries.shape,4).astype("float32")


def regular_labels(t, y, valid, sparse):
    """Interpolate supervision between nearby native labels or the known zero at decision.
    Sparse windows only supply structural labels on every fifth grid point.
    """
    b = len(t)
    q = np.arange(1, STRUCT_STEPS+1)*DT
    result = np.empty((b,STRUCT_STEPS,2,3),np.float32)
    eligible = np.empty((b,2),bool)
    rows = np.arange(b)[:,None]
    for j in range(2):
        times = np.concatenate([np.zeros((b,1)),np.where(valid[...,j],t[...,j],5.)],1)
        values = np.concatenate([np.zeros((b,1,3)),y[:,:,j]],1)
        ix = np.argsort(times,axis=1,kind="stable")
        times = times[rows,ix]
        values = values[rows,ix]
        base = rows * 10.
        hi = np.searchsorted((base+times).ravel(),(base+q).ravel(),"left").reshape(b,-1)-rows*times.shape[1]
        hi=hi.clip(1,times.shape[1]-1)
        lo=hi-1
        gap=times[rows,hi]-times[rows,lo]
        alpha=((q-times[rows,lo])/np.maximum(gap,1e-10)).clip(0,1)
        result[:,:,j]=values[rows,lo]*(1-alpha[...,None])+values[rows,hi]*alpha[...,None]
        good=(times[rows,hi]<5.) & (gap<=np.where(sparse,.16,.04)[:,None]+1e-6)
        eligible[:,j]=np.where(sparse,good[:,4::5].all(1),good.all(1))
    return result,eligible


def prepare(part):
    assert part in ("train","validation")
    dest = CACHE/part
    dest.mkdir(parents=True,exist_ok=True)
    ids = pd.read_parquet(FIELDS/part/"ids.parquet")
    window = pd.read_parquet(HIST/f"{part}_windows.parquet").set_index("window_id",verify_integrity=True)
    idx = window.index.get_indexer(ids.window_id)
    assert (idx>=0).all() and ids.window_id.is_unique
    w = window.iloc[idx]
    np.testing.assert_array_equal(w.flight_id.to_numpy(),ids.flight_id.to_numpy())
    tape_ids = pd.read_parquet(TAPE/f"{part}_ids.parquet")
    np.testing.assert_array_equal(tape_ids.window_id,ids.window_id)
    tape = np.load(TAPE/f"{part}_commands.npy",mmap_mode="r")
    raw = {key:np.load(HIST/f"{part}_{key}.npy",mmap_mode="r") for key in ("initial","t","valid","y")}
    fields = {key:np.load(FIELDS/part/f"{key}.npy",mmap_mode="r") for key in ("origin","origin_mask","time","mask","y")}
    assert fields["origin_mask"].all() and fields["mask"].all()
    sparse = w.sampling_group.to_numpy()=="sparse_10hz"
    np.save(dest/"sparse.npy",sparse)
    n=len(ids)
    K=raw["t"].shape[1]
    shapes={"origin":(2,3),"query_u":(8,2,4)}
    if part=="train":
        shapes.update(t=(K,2), valid=(K,2), y=(K,2,3), u=(K,2,4),
                      regular=(STRUCT_STEPS,2,3), regular_u=(STRUCT_STEPS,4), eligible=(2,))
    arr={k:np.lib.format.open_memmap(dest/f"{k}.npy","w+",dtype=bool if k in ("valid","eligible") else "float32",
                  shape=(n,*s)) for k,s in shapes.items()}
    sums=np.zeros(6);sq=np.zeros(6)
    endpoint_error=0.
    for start in range(0,n,256):
        sl=slice(start,min(start+256,n)); i=idx[sl]; b=len(i)
        init=raw["initial"][i]
        rot=Rotation.from_quat(init[:,[7,8,9,6]]).as_matrix()
        origin=np.stack([np.einsum("bi,bij->bj",init[:,3:6],rot),init[:,10:13]],1)
        arr["origin"][sl]=origin
        hist=fields["origin"][sl].astype("float64")+origin.reshape(b,6,1)
        sums+=hist.sum((0,2));sq+=np.square(hist).sum((0,2))
        query=fields["time"][sl]
        tape_part = tape[sl]
        end=tape_part[...,4].reshape(b,-1).sum(1)*1e-6
        assert (query.max((1,2))<=end+2e-6).all() and query.max() < H*DT
        arr["query_u"][sl]=integral(tape_part,query)
        if part=="validation": continue
        t=raw["t"][i][:,:,[0,2]]
        valid=raw["valid"][i][:,:,[0,2]].astype(bool) & (t>0) & (t<=end[:,None,None]+1e-6) & (t<=H*DT)
        t=np.where(valid,t,0.)
        values=raw["y"][i]
        y=np.stack([np.einsum("bki,bij->bkj",values[:,:,3:6]-init[:,None,3:6],rot),
                    values[:,:,10:13]-init[:,None,10:13]],2).astype("float32")
        y=np.where(valid[...,None],y,0.)
        for j in range(2):
            near=np.where(valid[:,None,:,j],abs(t[:,None,:,j]-query[:,:,j,None]),np.inf).argmin(2)
            ref=y[np.arange(b)[:,None],near,j]
            endpoint_error=max(endpoint_error,float(abs(ref-fields["y"][sl,:,j]).max()))
        arr["t"][sl]=t;arr["valid"][sl]=valid;arr["y"][sl]=y
        arr["u"][sl]=integral(tape_part,t)
        target,good=regular_labels(t,y,valid,sparse[sl])
        arr["regular"][sl]=target;arr["eligible"][sl]=good
        arr["regular_u"][sl]=integral(tape_part,np.broadcast_to(np.arange(1,STRUCT_STEPS+1)*DT,(b,STRUCT_STEPS)))
    if part=="train":
        assert endpoint_error<2e-4,endpoint_error
        mean=sums/(n*32);std=np.sqrt(sq/(n*32)-mean**2)
        assert (std>0).all()
        np.savez(CACHE/"scaling.npz",mean=mean,std=std)
    result=dict(windows=n,flights=ids.flight_id.nunique(),max_native_target_error=endpoint_error if part=="train" else None)
    if part=="train":
        result["structural_eligible_counts"]={g:{"windows":int(mask.sum()),"both_streams":int(arr["eligible"][mask].all(1).sum())} for g,mask in (("main",~sparse),("sparse",sparse))}
    write(dest/"ready.json",result)
    print(result,flush=True)


if __name__ == "__main__":
    for part in ("train","validation"):
        prepare(part)
