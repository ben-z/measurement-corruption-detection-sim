import numpy as np

def get_initial_state():
    return np.zeros(5)

def get_noop_action():
    return np.zeros(2)

def discrete_kinematic_bicycle_model(x, u, dt, L=2.9):
    # Kinematic bicycle model (rear axle reference frame, discrete time)
    # x = [x, y, theta, v, delta]
    # u = [a, delta_dot]
    x_dot = np.zeros(5)
    x_dot[0] = x[3] * np.cos(x[2])
    x_dot[1] = x[3] * np.sin(x[2])
    x_dot[2] = x[3] * np.tan(x[4]) / L
    x_dot[3] = u[0]
    x_dot[4] = u[1]
    x = x + x_dot * dt
    return x
