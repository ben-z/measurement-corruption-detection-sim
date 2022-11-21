from copy import deepcopy
import continuous_kinematic_bicycle as ckb
import control
import discrete_kinematic_bicycle as dkb
import numpy as np
from utils import closest_point_on_line, distance_to_line_segment, wrap_to_pi, generate_segment_info, move_along_path
from numpy.linalg import norm

TARGET_STEERING = 0 # rad
LOOKAHEAD_M = 5 # meters

def get_initial_state(dt, L, target_speed, Q=np.eye(5), R=np.eye(2)):
    return {
        'target_speed': target_speed, # m/s
        # time step
        'dt': dt,
        # wheelbase
        'L': L,
        # model
        # 'model': dkb.make_discrete_kinematic_bicycle_model(L, dt),
        'model': ckb.make_continuous_kinematic_bicycle_model(L),
        'Q': Q,
        'R': R,
    }

def lookahead_lqr(ext_state, estimate):
    """
    Path following controller using model predictive control.
    The control model is a discrete kinematic bicycle model.
    
    ext_state: controller state
    estimate: state estimate

    returns: (action, new_controller_state)

    action: [a, delta_dot]
    estimate: [x, y, theta, v, delta]
    """
    
    debug_output = {}
    
    # find the closest point on the path (assuming linear interpolation)
    target_path = ext_state['target_path']
    model = ext_state['model']
    # x, y, theta, v, delta = estimate
    x = estimate[0]
    y = estimate[1]

    pos = np.array([x,y])

    segment_info = generate_segment_info(pos, target_path, wrap=False)
    current_path_segment_idx = np.argmin([norm(info.closest_point - pos) for info in segment_info])

    lookahead_path_segment_info, _ = next(move_along_path(deepcopy(segment_info), current_path_segment_idx, LOOKAHEAD_M))

    debug_output['current_path_segment'] = segment_info[current_path_segment_idx].__dict__
    debug_output['lookahead_path_segment'] = lookahead_path_segment_info.__dict__

    # Linearize, assuming the reference is a line. (See 2022-09-15 and 2022-09-22 notes for derivations)
    linearization_state = np.array(
        [0, 0, lookahead_path_segment_info.heading, ext_state['target_speed'], 0])
    linearization_input = np.zeros(model.ninputs)
    linsys = model.linearize(linearization_state, linearization_input)
    # linsys = model
    linsysd = linsys.sample(ext_state['dt'])

    # compute the desired state
    target_x = np.zeros(estimate.shape)
    target_x[:2] = lookahead_path_segment_info.p0 + \
        (lookahead_path_segment_info.p1 - lookahead_path_segment_info.p0) * \
        lookahead_path_segment_info.progress
    target_x[2] = lookahead_path_segment_info.heading
    target_x[3] = ext_state['target_speed']
    target_x[4] = TARGET_STEERING

    assert target_x.shape == estimate.shape, f"This controller only supports estimates of shape {target_x.shape}, got {estimate.shape}"

    debug_output['target_x'] = [target_x]

    # compute the LQR gain
    K, _S, _E = control.dlqr(linsysd, ext_state["Q"], ext_state["R"])

    diff = estimate - target_x
    diff[2] = wrap_to_pi(diff[2])

    u = - K @ diff

    debug_output['K'] = K

    return u, ext_state, debug_output
