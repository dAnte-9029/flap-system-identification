"""Empirical logged-prediction error references, not counterfactual guarantees."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ActionMargin:
    nominal_improvement: float
    two_prediction_error_allowance: float
    residual_margin: float
    exceeds_empirical_allowance: bool
    control_validated: bool = False


class EmpiricalErrorEnvelope:
    def __init__(self,rows):
        self._bounds={}
        for row in rows:
            key=(int(row['steps']),str(row['axis']))
            value=float(row['absolute_error_bound'])
            if key in self._bounds or not math.isfinite(value) or value<=0:
                raise ValueError('invalid or duplicate empirical bound')
            self._bounds[key]=value
        if not self._bounds:raise ValueError('empty empirical envelope')

    def bound(self,steps,axis):
        """No extrapolation/interpolation outside evaluated native-step horizons."""
        if not isinstance(steps,int):raise ValueError('integer native-step horizon required')
        try:return self._bounds[(steps,axis)]
        except KeyError as e:raise ValueError('unassessed horizon/axis') from e

    def action_margin(self,hold_prediction,action_prediction,target,*,steps,axis):
        values=(hold_prediction,action_prediction,target)
        if not all(math.isfinite(v) for v in values):raise ValueError('nonfinite forecast or target')
        nominal=abs(hold_prediction-target)-abs(action_prediction-target)
        allowance=2*self.bound(steps,axis)
        margin=nominal-allowance
        # Even a positive margin does not establish counterfactual error coverage.
        return ActionMargin(nominal,allowance,margin,margin>0,False)
