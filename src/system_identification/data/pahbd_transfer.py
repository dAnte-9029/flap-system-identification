"""CPU-only native loading and one packed device transfer per ragged window."""
from dataclasses import fields,replace
import copy
import torch
from .pahbd import ConfiguredWindows


def cpu_windows(windows):
    if not isinstance(windows,ConfiguredWindows):raise TypeError('prefetch requires native ConfiguredWindows')
    dataset=copy.copy(windows.dataset)
    dataset.device='cpu';dataset._cached_id=None;dataset.tables={}
    return ConfiguredWindows(dataset,windows.indices,windows.config)


def window_to_device(window,device):
    """Preserve native times, masks and metadata; no interpolation or padding."""
    objects=[window.initial,*window.history,*window.targets,window.controls]
    slots=[(i,f.name,getattr(obj,f.name)) for i,obj in enumerate(objects) for f in fields(obj)
           if isinstance(getattr(obj,f.name),torch.Tensor)]
    if any(t.device.type!='cpu' or t.requires_grad for _,_,t in slots):
        raise ValueError('packed data transfer requires non-differentiable CPU observations/controls')
    packed=torch.cat([t.reshape(-1) for _,_,t in slots]).to(device)
    updates=[{} for _ in objects];offset=0
    for i,name,t in slots:
        updates[i][name]=packed[offset:offset+t.numel()].view(t.shape);offset+=t.numel()
    moved=[replace(obj,**update) for obj,update in zip(objects,updates)]
    nh=len(window.history);tape=moved[-1]
    original=window.controls.canonical()
    # CPU canonicalization is already done by the native loader.
    if original.times!=tape.times:raise ValueError('control tape must be canonical before packing')
    tape._cpu_values=original._cpu_values
    tape._prepared_key=(id(tape.values),tape.values._version,tuple(tape.times),False)
    return replace(window,initial=moved[0],history=moved[1:1+nh],targets=moved[1+nh:-1],controls=tape)
