import control
import numpy as np

def get_initial_state():
    return np.zeros(5)

def get_noop_action():
    return np.zeros(2)

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


def make_discrete_kinematic_bicycle_model(L=2.9, dt=0.01):
    return control.NonlinearIOSystem(
        discrete_kinematic_bicycle_model,
        inputs=('a', 'delta_dot'),
        outputs=('x', 'y', 'theta', 'v', 'delta'),
        states=('x', 'y', 'theta', 'v', 'delta'),
        params={'L': L, 'dt': dt},
        dt=dt,
    )
