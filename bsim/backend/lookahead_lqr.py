import continuous_kinematic_bicycle as ckb
import control
import discrete_kinematic_bicycle as dkb
import numpy as np
from utils import closest_point_on_line, distance_to_line_segment, wrap_to_pi

TARGET_STEERING = 0 # rad
LOOKAHEAD_M = 5 # meters

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

    # pairs of points that form the path: [(p0, p1), (p1, p2), ..., (pn, p0)]
    path_segments = np.stack([target_path, np.roll(target_path, -1, axis=0)], axis=1)

    # segment_info is used to make decisions about which segment to use
    segment_info = []
    for p0, p1 in path_segments:
        segment_length = np.linalg.norm(p1 - p0)
        closest_point, progress = closest_point_on_line(np.array([x, y]), p0, p1)
        if progress < 0:
            closest_point = p0
            distance_travelled = 0
        elif progress > 1:
            closest_point = p1
            distance_travelled = 0
        else:
            closest_point = closest_point
            distance_travelled = progress * segment_length
        
        segment_info.append({
            'p0': p0,
            'p1': p1,
            'length': segment_length,
            'heading': np.arctan2(p1[1] - p0[1], p1[0] - p0[0]),
            'closest_point': closest_point,
            'progress': progress,
            'distance_from_ego': np.linalg.norm(closest_point - np.array([x, y])),
            'distance_travelled': distance_travelled,
            'distance_remaining': segment_length - distance_travelled,
        })
    
    current_path_segment_idx = np.argmin([info['distance_from_ego'] for info in segment_info])
    lookahead_path_segment_idx = current_path_segment_idx
    remaining_lookahead_m = LOOKAHEAD_M - segment_info[current_path_segment_idx]['distance_remaining']
    # find the segment that contains the lookahead point
    while remaining_lookahead_m > 0:
        lookahead_path_segment_idx = (lookahead_path_segment_idx + 1) % len(segment_info)
        remaining_lookahead_m -= segment_info[lookahead_path_segment_idx]['distance_remaining']

    # remaining_lookahead_m is now the distance from the end of the lookahead segment to the lookahead point
    # and is always non-positive
    target_progress = 1 + remaining_lookahead_m / segment_info[lookahead_path_segment_idx]['length']

    lookahead_path_segment_info = segment_info[lookahead_path_segment_idx]

    debug_output['current_path_segment'] = path_segments[current_path_segment_idx]
    debug_output['lookahead_path_segment'] = path_segments[lookahead_path_segment_idx]

    # Linearize, assuming the reference is a line. (See 2022-09-15 and 2022-09-22 notes for derivations)
    linearization_state = np.array(
        [0, 0, lookahead_path_segment_info['heading'], state['target_speed'], 0])
    linearization_input = np.zeros(model.ninputs)
    linsys = model.linearize(linearization_state, linearization_input)
    # linsys = model
    linsysd = linsys.sample(state['dt'])

    # compute the desired state
    target_x = np.zeros(estimate.shape)
    target_x[:2] = lookahead_path_segment_info['p0'] + (lookahead_path_segment_info['p1'] - lookahead_path_segment_info['p0']) * target_progress
    target_x[2] = lookahead_path_segment_info['heading']
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
