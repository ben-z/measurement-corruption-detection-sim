import control
import math
import numpy as np
from utils import wrap_to_pi


_INITIAL_STATE = np.array([0, 0, 0, 0.0001, 0])
# OUTPUTS = ('x', 'y', 'theta', 'v', 'delta')
# OUTPUTS = ('x', 'y', 'theta', 'v', 'delta', 'x1', 'y1')
# OUTPUTS = ('x', 'y', 'theta', 'v', 'delta', 'x1', 'y1', 'v1')
OUTPUTS = ('x', 'y', 'theta', 'v', 'delta', 'x1', 'y1', 'theta1', 'v1')
# OUTPUTS = ('x', 'y', 'theta', 'v', 'delta', 'x1', 'y1', 'theta1', 'v1', 'delta1')
# OUTPUTS = ('x', 'y', 'theta', 'v', 'delta', 'x1', 'y1', 'theta1', 'v1', 'delta1', 'x2', 'y2', 'theta2', 'v2', 'delta2')
STATES = ('x', 'y', 'theta', 'v', 'delta')
INPUTS = ('a', 'delta_dot')

assert len(STATES) == _INITIAL_STATE.shape[0]

def get_initial_state():
    return np.copy(_INITIAL_STATE)


def get_noop_action():
    return np.zeros(2)


def normalize_state(state):
    state[get_angular_states_mask()] = wrap_to_pi(state[get_angular_states_mask()])
    return state

_ANGULAR_STATES_MASK = np.array([s.startswith(('theta', 'delta')) for s in STATES])
def get_angular_states_mask():
    return _ANGULAR_STATES_MASK

_ANGULAR_OUTPUTS_MASK = np.array([s.startswith(('theta', 'delta')) for s in OUTPUTS])
def get_angular_outputs_mask():
    return _ANGULAR_OUTPUTS_MASK
    

def continuous_kinematic_bicycle_model(t, x, u, params):
    # Kinematic bicycle model (rear axle reference frame, continuous time)
    # x = [x, y, theta, v, delta]
    # u = [a, delta_dot]

    L = 2.9
    if 'L' in params:
        L = params['L']

    x_dot = np.zeros(len(STATES))
    assert len(STATES) == 5
    x_dot[0] = x[3] * np.cos(x[2])
    x_dot[1] = x[3] * np.sin(x[2])
    x_dot[2] = x[3] * np.tan(x[4]) / L
    x_dot[3] = u[0]
    x_dot[4] = u[1]

    return x_dot


def continuous_kinematic_bicycle_model_output(t, x, u, params):
    y = np.zeros(len(OUTPUTS))

    y[0:5] = x[0:5]
    # additional sensor
    y[5:9] = x[0:4] 
    # y[7] = x[3] 
    # y[5:10] = x[0:5] 
    # y[10:15] = x[0:5] 

    return y

def make_continuous_kinematic_bicycle_model(L=2.9):
    return control.NonlinearIOSystem(
        continuous_kinematic_bicycle_model,
        outfcn=continuous_kinematic_bicycle_model_output,
        inputs=INPUTS,
        outputs=OUTPUTS,
        states=STATES,
        params={'L': L},
    )

def make_model(*args, **kwargs):
    return make_continuous_kinematic_bicycle_model(*args, **kwargs)

def _get_C():
    """
    Returns the output matrix C for the linearized model.
    This currently only supports returning states without arithmetic.
    Can extend to support arithmetic by using bit masks.
    """
    x = np.arange(len(STATES))
    y = continuous_kinematic_bicycle_model_output(0, x, None, None)

    C = np.zeros((len(OUTPUTS), len(STATES)))

    for i in range(len(OUTPUTS)):
        for j in range(len(STATES)):
            C[i,j] = 1 if y[i] == x[j] else 0
    
    return C

_C = _get_C()

def get_linear_model_straight_line_ref(x,y,theta,v,delta,a,delta_dot,L):
    A = np.array([
        [0, 0, -v*np.sin(theta), np.cos(theta), 0],
        [0, 0, v*np.cos(theta), np.sin(theta), 0],
        [0, 0, 0, 0, v/L],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])

    B = np.array([
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 1],
    ])

    C = _C.copy()

    D = np.zeros((len(OUTPUTS), len(INPUTS)))

    return A,B,C,D

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
