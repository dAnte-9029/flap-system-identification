import torch
from torch.torch_version import TorchVersion
from test_pahbd_readiness import model_window
from system_identification.training.pahbd import save_checkpoint,load_checkpoint


def test_load_legacy_torch_version_metadata_and_adam(tmp_path):
    model,_=model_window(.02)
    opt=torch.optim.Adam(model.parameters())
    sum(p.square().sum() for p in model.parameters()).backward();opt.step()
    path=tmp_path/'saved.pt'
    save_checkpoint(path,model,opt,{'runtime':{'torch':TorchVersion('2.7.0+cu128')}},1)
    loaded,restored,bundle=load_checkpoint(path,optimizer=True)
    assert bundle['step']==1 and str(bundle['metadata']['runtime']['torch'])=='2.7.0+cu128'
    for k,v in model.state_dict().items():torch.testing.assert_close(v,loaded.state_dict()[k],rtol=0,atol=0)
    for k,state in opt.state_dict()['state'].items():
        for name,value in state.items():torch.testing.assert_close(value,restored.state_dict()['state'][k][name],rtol=0,atol=0)
