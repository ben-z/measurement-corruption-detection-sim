import numpy as np
import discrete_kinematic_bicycle as dkb

def get_initial_state():
    return np.zeros(5)

def path_following_kmpc(state, estimate):
    """
    Path following controller using model predictive control.
    The control model is a discrete kinematic bicycle model.
    
    state: controller state
    estimate: state estimate

    returns: (action, new_controller_state)

    action: [a, delta_dot]
    estimate: [x, y, theta, v, delta]
    """
    
    # TODO: write an actual MPC
    
    v = 5
    delta_dot = -0.5
    
    if estimate[3] > 10:
        v = 0
    
    if abs(estimate[4]) > 0.5:
        delta_dot = 0
    
    return np.array([v, delta_dot]), state
