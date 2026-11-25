"""Export all eight native predictions and identifiers for transfer of an independently frozen head."""
import argparse

import numpy as np
import torch

from tcn import BASE, BATCH, HERE, SUITE, configure, data, hydrate, inverse, json, query, rotate


@torch.inference_mode()
def export(name):
    configure()
    ck, model = hydrate(SUITE/'tcn/best.pt')
    if name!='archived':
        model.load_state_dict(torch.load(HERE/(name+'.pt'),weights_only=True,map_location='cpu'))
    model.eval()
    ds, vid, audit = data('validation')
    static = [v[:BATCH].clone() for v in ds]
    def predict():
        hist, initial, tape, t, _, _ = static
        roll=model(hist,initial,tape)
        velocity=query(roll,t[...,0],'pv')[...,3:]
        rate=query(roll,t[...,1],'w')
        dv=rotate(inverse(initial[:,None,6:10]),velocity-initial[:,None,3:6])
        dw=rate-initial[:,None,10:13]
        return torch.stack((dv,dw),2)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        predict()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = predict()
    predictions = []
    for start in range(0,len(vid),BATCH):
        stop=min(len(vid),start+BATCH)
        for dest,source in zip(static,ds):
            dest[:stop-start].copy_(source[start:stop])
        graph.replay()
        predictions.append(output[:stop-start].cpu().numpy())
    yhat=np.concatenate(predictions)
    with np.load(BASE/'validation.npz') as src:
        flight,window,time,target=src['flight'],src['window'],src['time'],src['y']
    assert np.array_equal(flight,vid)
    np.savez_compressed(HERE/f'predictions_seed17_{name}.npz',predicted_delta=yhat,
                        window_id=window,flight_id=flight,time=time)
    scores=json.loads((HERE/(name+'_flight_mse.json')).read_text())
    sq=(yhat-target)**2
    mse=sq.sum(-1)[:,[1,3,7]].reshape(-1,6)
    maxdiff=max(abs(float(mse[flight==fid,j].astype('float64').mean())-scores[key][fid])
               for j,key in enumerate(('v05','w05','v1','w1','v2','w2')) for fid in scores[key])
    assert maxdiff<2e-5, maxdiff
    print(name,yhat.shape,'max flight MSE discrepancy',maxdiff,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['archived','best','terminal'])
    export(parser.parse_args().stage)
