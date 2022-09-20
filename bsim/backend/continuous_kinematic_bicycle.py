import control
import math
import numpy as np
from discrete_kinematic_bicycle import normalize_state, get_noop_action

_INITIAL_STATE = np.array([0, 0, 0, 0.0001, 0])

def get_initial_state():
    return np.copy(_INITIAL_STATE)

def continuous_kinematic_bicycle_model(t, x, u, params):
    # Kinematic bicycle model (rear axle reference frame, continuous time)
    # x = [x, y, theta, v, delta]
    # u = [a, delta_dot]

    L = 2.9
    if 'L' in params:
        L = params['L']

    x_dot = np.zeros(_INITIAL_STATE.shape)
    x_dot[0] = x[3] * np.cos(x[2])
    x_dot[1] = x[3] * np.sin(x[2])
    x_dot[2] = x[3] * np.tan(x[4]) / L
    x_dot[3] = u[0]
    x_dot[4] = u[1]

    return x_dot


def continuous_kinematic_bicycle_model_output(t, x, u, params):
    y = np.zeros(6)

    y[0:5] = x[0:5]
    y[5] = x[0] # additional sensor for x

    return y

def make_continuous_kinematic_bicycle_model(L=2.9):
    return control.NonlinearIOSystem(
        continuous_kinematic_bicycle_model,
        outfcn=continuous_kinematic_bicycle_model_output,
        inputs=('a', 'delta_dot'),
        # outputs=('x', 'y', 'theta', 'v', 'delta'),
        outputs=('x', 'y', 'theta', 'v', 'delta', 'x1'),
        states=('x', 'y', 'theta', 'v', 'delta'),
        params={'L': L},
    )

def make_model(*args, **kwargs):
    return make_continuous_kinematic_bicycle_model(*args, **kwargs)

if __name__ == '__main__':
    sys = make_continuous_kinematic_bicycle_model()
    print(sys)

    # compare manually linearized model with auto-linearized model
    # state = np.array([0, 0, 0, 0.001, 0.5])
    state = np.random.random(5) * 10
    u = np.random.random(2)
    print("state")
    print(state)
    print("u")
    print(u)
    theta = state[2]
    v = state[3]
    delta = state[4]
    L = 2.9
    A = np.array([
        [0, 0, -v*math.sin(theta), math.cos(theta), 0],
        [0, 0, v*math.cos(theta), math.sin(theta), 0],
        [0, 0, 0, math.tan(delta)/L, v/(L*math.cos(delta)**2)],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]])
    B = np.array([
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 1]])
    
    linsys = sys.linearize(state, u)
    print("A_diff")
    print((linsys.A - A).round(decimals=5))
    print("B_diff")
    print((linsys.B - B).round(decimals=5))

    dt = 0.01
    linsysd = linsys.sample(dt)

    K_auto, S_auto, E_auto = control.dlqr(linsysd, np.eye(5), np.eye(2))
    K, S, E = control.dlqr(linsysd, np.eye(5), np.eye(2))

    print("K_diff")
    print((K_auto - K).round(decimals=5))
    print("S_diff")
    print((S_auto - S).round(decimals=5))
    print("E_diff")
    print((E_auto - E).round(decimals=5))

    pass
