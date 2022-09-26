import continuous_kinematic_bicycle as ckb
import control
import discrete_kinematic_bicycle as dkb
import numpy as np
from utils import closest_point_on_line, distance_to_line_segment, wrap_to_pi

def get_initial_state(target_path, dt, L, target_speed, Q=np.eye(5), R=np.eye(2)):
    return {
        # coordinates of the waypoints in meters, this will form a closed path
        'target_path': target_path,
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

def lookahead_lqr(state, estimate):
    """
    Path following controller using model predictive control.
    The control model is a discrete kinematic bicycle model.
    
    state: controller state
    estimate: state estimate

    returns: (action, new_controller_state)

    action: [a, delta_dot]
    estimate: [x, y, theta, v, delta]
    """
    
    debug_output = {}
    
    # find the closest point on the path (assuming linear interpolation)
    target_path = state['target_path']
    model = state['model']
    # x, y, theta, v, delta = estimate
    x = estimate[0]
    y = estimate[1]

    # Linearize, assuming the reference is a line. (See 2022-09-15 and 2022-09-22 notes for derivations)
    linearization_state = estimate.copy()
    linearization_state[4] = 0
    linearization_input = np.zeros(model.ninputs)
    linsys = model.linearize(linearization_state, linearization_input)
    # linsys = model
    linsysd = linsys.sample(state['dt'])

    path_segments = np.stack([target_path, np.roll(target_path, -1, axis=0)], axis=1)

    current_path_segment_index = np.argmin([
        distance_to_line_segment(np.array([x, y]), p1, p2) for p1, p2 in path_segments
    ])

    current_path_segment = path_segments[current_path_segment_index]
    current_path_heading = np.arctan2(current_path_segment[1, 1] - current_path_segment[0, 1], current_path_segment[1, 0] - current_path_segment[0, 0])
    current_path_segment_length = np.linalg.norm(current_path_segment[1] - current_path_segment[0])

    debug_output['current_path_segment'] = current_path_segment
    
    TARGET_STEERING = 0 # rad
    LOOKAHEAD_M = 5 # meters

    # compute the desired states along the path
    _p_closest, progress = closest_point_on_line(np.array([x, y]), current_path_segment[0], current_path_segment[1])

    target_progress = progress + LOOKAHEAD_M / current_path_segment_length
    target_x = np.zeros(estimate.shape)
    target_x[:2] = current_path_segment[0] + (current_path_segment[1] - current_path_segment[0]) * target_progress
    target_x[2] = current_path_heading
    target_x[3] = state['target_speed']
    target_x[4] = TARGET_STEERING

    assert target_x.shape == estimate.shape, f"This controller only supports estimates of shape {target_x.shape}, got {estimate.shape}"

    debug_output['target_x'] = [target_x]

    # compute the LQR gain
    K, _S, _E = control.dlqr(linsysd, state["Q"], state["R"])

    diff = estimate - target_x
    diff[2] = wrap_to_pi(diff[2])

    u = - K @ diff

    debug_output['K'] = K

    return u, state, debug_output
