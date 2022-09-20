import control
import numpy as np

from utils import wrap_to_pi

def get_initial_state():
    return np.array([0, 0, 0, 0.0001, 0])

def get_noop_action():
    return np.zeros(2)

def normalize_state(state):
    state[2] = wrap_to_pi(state[2])
    state[4] = wrap_to_pi(state[4])
    return state


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

    x_dot = np.zeros(5)
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


def discrete_kinematic_bicycle_model_output(t, x, u, params):
    y = np.zeros(6)

    y[0:5] = x[0:5]
    y[5] = x[0]  # additional sensor for x

    return y

def make_discrete_kinematic_bicycle_model(L=2.9, dt=0.01):
    return control.NonlinearIOSystem(
        discrete_kinematic_bicycle_model,
        outfcn=discrete_kinematic_bicycle_model_output,
        inputs=('a', 'delta_dot'),
        # outputs=('x', 'y', 'theta', 'v', 'delta'),
        outputs=('x', 'y', 'theta', 'v', 'delta', 'x1'),
        states=('x', 'y', 'theta', 'v', 'delta'),
        params={'L': L, 'dt': dt},
        dt=dt,
    )


def make_model(*args, **kwargs):
    return make_discrete_kinematic_bicycle_model(*args, **kwargs)