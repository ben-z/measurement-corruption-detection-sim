import control
import numpy as np
from continuous_kinematic_bicycle import INPUTS, STATES, OUTPUTS, continuous_kinematic_bicycle_model_output, get_noop_action, normalize_state

def get_initial_state():
    return np.array([0, 0, 0, 0.0001, 0])

def discrete_kinematic_bicycle_model(t, x, u, params):
    # Kinematic bicycle model (rear axle reference frame, discrete time)
    # x = [x, y, theta, v, delta]
    # u = [a, delta_dot]
    
    dt = 0.01
    L = 2.9
    if 'L' in params:
        L = params['L']
    if 'dt' in params:
        dt = params['dt']

    assert len(STATES) == 5
    x_dot = np.zeros(len(STATES))
    x_dot[0] = x[3] * np.cos(x[2])
    x_dot[1] = x[3] * np.sin(x[2])
    x_dot[2] = x[3] * np.tan(x[4]) / L
    x_dot[3] = u[0]
    x_dot[4] = u[1]
    x = x + x_dot * dt
    
    # Actuator constraints
    if x[4] > np.pi / 4:
        x[4] = np.pi / 4
    elif x[4] < -np.pi / 4:
        x[4] = -np.pi / 4

    return x


def discrete_kinematic_bicycle_model_output(*args, **kwargs):
    return continuous_kinematic_bicycle_model_output(*args, **kwargs)

def make_discrete_kinematic_bicycle_model(L=2.9, dt=0.01):
    return control.NonlinearIOSystem(
        discrete_kinematic_bicycle_model,
        outfcn=discrete_kinematic_bicycle_model_output,
        inputs=INPUTS,
        outputs=OUTPUTS,
        states=STATES,
        params={'L': L, 'dt': dt},
        dt=dt,
    )


def make_model(*args, **kwargs):
    return make_discrete_kinematic_bicycle_model(*args, **kwargs)