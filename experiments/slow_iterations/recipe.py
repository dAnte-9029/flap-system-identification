"""The stage-specific model and losses. Edited only after the preceding round is scored."""
from shared import Dynamics
ARMS={'baseline':dict(coupled=False,feedback=False,consistency=False)}

def build(original,arm):
    net=Dynamics(original,arm['coupled'])
    if net.coupled and not arm['feedback']:
        net.feedback.weight.requires_grad_(False)
    return net

def extra_loss(model,arm,outputs):
    return outputs['consistency'] if arm.get('consistency') else 0.
