from os import curdir
import numpy as np
import discrete_kinematic_bicycle as dkb
import continuous_kinematic_bicycle as ckb
import scipy

def get_initial_state(target_path, dt, L):
    return {
        # coordinates of the waypoints in meters, this will form a closed path
        'target_path': target_path,
        # time step
        'dt': dt,
        # wheelbase
        'L': L,
        # model
        'model': dkb.make_discrete_kinematic_bicycle_model(L, dt),
    }

def closest_point_on_line_segment(p, a, b):
    """
    Find the closest point on the line segment defined by a and b to the point p.
    Also returns the parameter t, which is the progress (0-1) along the line segment ab.
    """
    # https://stackoverflow.com/a/1501725
    ap = p - a
    ab = b - a
    ab2 = np.dot(ab, ab)
    ap_ab = np.dot(ap, ab)
    t = ap_ab / ab2
    if t < 0:
        return a, 0
    elif t > 1:
        return b, 1
    else:
        return a + ab * t, t

def distance_to_line_segment(p, a, b):
    """
    Find the distance from the point p to the line segment defined by a and b.
    """
    return np.linalg.norm(p - closest_point_on_line_segment(p, a, b)[0])


def closest_point_on_line(p, a, b):
    """
    Find the closest point on the line defined by a and b to the point p.
    Also returns the parameter t, which is the progress (0-1) along the line ab.
    """
    # https://stackoverflow.com/a/1501725
    ap = p - a
    ab = b - a
    ab2 = np.dot(ab, ab)
    ap_ab = np.dot(ap, ab)
    t = ap_ab / ab2
    return a + ab * t, t

def distance_to_line(p, a, b):
    """
    Find the distance from the point p to the line defined by a and b.
    """
    return np.linalg.norm(p - closest_point_on_line(p, a, b)[0])

def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

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
    debug_output = {}
    
    # find the closest point on the path (assuming linear interpolation)
    target_path = state['target_path']
    model = state['model']
    x, y, theta, v, delta = estimate

    # Control performance after linearization is very poor
    linsys = model.linearize(estimate, [0, 0])
    # linsys = model

    path_segments = np.stack([target_path, np.roll(target_path, -1, axis=0)], axis=1)

    current_path_segment_index = np.argmin([
        distance_to_line_segment(np.array([x, y]), p1, p2) for p1, p2 in path_segments
    ])

    current_path_segment = path_segments[current_path_segment_index]
    current_path_heading = np.arctan2(current_path_segment[1, 1] - current_path_segment[0, 1], current_path_segment[1, 0] - current_path_segment[0, 0])
    current_path_segment_length = np.linalg.norm(current_path_segment[1] - current_path_segment[0])

    debug_output['current_path_segment'] = current_path_segment
    
    M = 5
    N = 40
    NUM_ACTION_VARS = 2
    TARGET_SPEED = 1 # m/s
    TARGET_STEERING = 0 # rad

    # compute the desired states along the path
    _p_closest, progress = closest_point_on_line(np.array([x, y]), current_path_segment[0], current_path_segment[1])
    target_x = np.zeros([N, estimate.shape[0]])
    # current_path_segment_length 
    target_progress = progress + (np.linspace(0, TARGET_SPEED * state['dt'] * N, N) / current_path_segment_length)
    target_x[:, :2] = current_path_segment[0] + (current_path_segment[1] - current_path_segment[0]) * target_progress[:, None]
    target_x[:, 2] = current_path_heading
    target_x[:, 3] = TARGET_SPEED
    target_x[:, 4] = TARGET_STEERING

    debug_output['target_x'] = target_x

    u_initial_guess = np.zeros((M, NUM_ACTION_VARS))
    u_bounds = np.zeros((M, NUM_ACTION_VARS, 2))
    u_bounds[:, 0, :] = [0, 10]
    u_bounds[:, 1, :] = [-0.5, 0.5]
    
    def predict_trajectory(update_fn, x0, u):
        x = np.zeros((u.shape[0], x0.shape[0]))
        x[0] = x0
        for i in range(x.shape[0]-1):
            x[i+1] = update_fn(x[i], u[i])
        return x
    
    def objective(u_arr):
        """
        Objective function for the MPC.
        """
        u_arr = u_arr.reshape((M, NUM_ACTION_VARS))
        u_arr = np.concatenate((u_arr, np.broadcast_to(u_arr[-1], (N - M, NUM_ACTION_VARS))))

        x_arr = predict_trajectory(lambda x, u: linsys.dynamics(0, x, u), estimate, u_arr)
        # x_arr = predict_trajectory(lambda x, u: dkb.discrete_kinematic_bicycle_model(x, u, state['dt'], state['L']), estimate, u_arr)

        distances = target_x[:, :2] - x_arr[:, :2]
        
        final_heading_diff = wrap_to_pi(current_path_heading - x_arr[-1, 2])
        
        return np.sum(distances) + np.linalg.norm(x_arr[:, 3] - TARGET_SPEED)*10 + abs(final_heading_diff)*10
    
    result = scipy.optimize.minimize(objective, u_initial_guess.reshape((M*NUM_ACTION_VARS,)), bounds=u_bounds.reshape((M*NUM_ACTION_VARS,2)))
    u_opt_arr = result.x.reshape((M, NUM_ACTION_VARS))
    
    predicted_x = predict_trajectory(lambda x, u: linsys.dynamics(0, x, u), estimate, np.concatenate((u_opt_arr, np.broadcast_to(u_opt_arr[-1], (N - M, NUM_ACTION_VARS)))))
    # predicted_x = predict_trajectory(lambda x, u: dkb.discrete_kinematic_bicycle_model(x, u, state['dt'], state['L']), estimate, np.concatenate((u_opt_arr, np.broadcast_to(u_opt_arr[-1], (N - M, NUM_ACTION_VARS)))))
    debug_output['predicted_x'] = predicted_x
    final_heading = predicted_x[-1, 2]
    final_heading_diff = wrap_to_pi(current_path_heading - final_heading)
    
    return u_opt_arr[0], state, debug_output
