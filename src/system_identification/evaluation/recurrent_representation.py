"""Explicit oracle interventions for frozen-model failure decomposition only."""
from dataclasses import replace
from system_identification.models.rolling_history_simulator import state_features

PHYSICAL = ('position_n','velocity_n','quaternion_nb','angular_velocity_b',
            'relative_phase_rad','flap_frequency_hz')


def teacher_physical(state, truth_now):
    """NOT DEPLOYABLE. Proxies and fixed episode phase anchor are not reset."""
    return replace(state, **{k:truth_now[k] for k in PHYSICAL})


def teacher_recurrent_step(simulator, state, command, dt, truth_now, truth_next):
    """B: output at true x_t; G updates with true x_(t+1), never predicted x."""
    current=teacher_physical(state,truth_now)
    prediction,diagnostic=simulator.step(current,command,dt)
    next_teacher=teacher_physical(prediction,truth_next)
    base=simulator.model.base_model
    h=base.recurrent_cell(base._model_input(state_features(next_teacher),command),state.gru_hidden)
    return replace(next_teacher,gru_hidden=h),prediction,diagnostic
