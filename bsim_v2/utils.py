import cvxpy as cp
import numpy as np
import time
from itertools import chain, combinations
from math import pi, sin, cos, atan2, sqrt

#################################################################
# General utility functions
#################################################################

def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def clamp(x, lower, upper):
    return np.maximum(lower, np.minimum(x, upper))

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

#################################################################
# Models
#################################################################

# Kinematic bicycle model
def kinematic_bicycle_model(state, input, params):
    # state = [x, y, theta, v, delta]
    # input = [a, delta_dot]
    # params = {dt, l}
    dt = params['dt']
    l = params['l']

    x = state[0]
    y = state[1]
    theta = state[2]
    v = state[3]
    delta = state[4]

    a = input[0]
    delta_dot = input[1]

    x_dot = v * np.cos(theta)
    y_dot = v * np.sin(theta)
    theta_dot = v / l * np.tan(delta)

    x += x_dot * dt
    y += y_dot * dt
    theta += theta_dot * dt
    v += a * dt
    delta += delta_dot * dt

    return np.array([x, y, theta, v, delta])

#################################################################
# Paths
#################################################################

# Derived from research-jackal
def generate_circle_approximation(center, radius, num_points):
    """
    Generates a list of points on a circle.
    Arguments:
        center: The center of the circle.
        radius: The radius of the circle.
        num_points: The number of points to generate. The more points, the more accurate the approximation.
    Returns:
        points: A list of points on the figure eight shape.
        headings: A list of headings (θ) at each point.
        curvatures: A list of curvatures (κ) at each point.
        dK_ds_list: A list of dκ_ds at each point. Where κ is the curvature and s is the arc length.
    
    Derivation: https://github.com/ben-z/research-sensor-attack/blob/e20c7b02cf6aca6c18c37976550c03606919192a/curves.py#L173-L191
    """
    a = radius

    points = []
    headings = []
    curvatures = []
    dK_ds_list = []
    for i in range(num_points):
        t = 2 * pi * i / num_points
        points.append([
            center[0] + a * cos(t),
            center[1] + a * sin(t)
        ])
        headings.append(atan2(a*cos(t), -a*sin(t)))
        # This could be simplified to 1/a, but we leave it as is for consistency with the derivation.
        curvatures.append(1/sqrt(a**2*sin(t)**2 + a**2*cos(t)**2))
        dK_ds_list.append(0) # circles have constant curvature
    return points, headings, curvatures, dK_ds_list

# Derived from research-jackal
def generate_figure_eight_approximation(center, length, width, num_points):
    """
    Generates a list of points on a figure eight shape.
    Arguments:
        center: The center of the figure eight shape.
        length: The length of the figure eight shape.
        width: The width of the figure eight shape.
        num_points: The number of points to generate. The more points, the more accurate the approximation.
    Returns:
        points: A list of points on the figure eight shape.
        headings: A list of headings (θ) at each point.
        curvatures: A list of curvatures (κ) at each point.
        dK_ds_list: A list of dκ_ds at each point. Where κ is the curvature and s is the arc length.

    The formula used to generate the points is:
        x = a * sin(t)
        y = b * sin(2t)/2
    where a = length / 2 and b = width.

    Supplementary visualization:
    https://www.desmos.com/calculator/fciqxay3p2
    Derivation:
    https://github.com/ben-z/research-sensor-attack/blob/e20c7b02cf6aca6c18c37976550c03606919192a/curves.py#L153-L171
    """
    a = length / 2
    b = width

    points = []
    headings = []
    curvatures = []
    dK_ds_list = []
    for i in range(num_points):
        # t is an arbitrary parameter that is used to generate the points
        # The result is known as an arbitrary-speed curve
        t = 2 * pi * i / num_points
        x = center[0] + a * sin(t)
        y = center[1] + b * (sin(t * 2) / 2)
        points.append([x, y])
        headings.append(atan2(b * cos(t * 2), a * cos(t)))
        curvatures.append((a*b*sin(t)*cos(2*t) - 2*a*b*sin(2*t)*cos(t))/(a**2*cos(t)**2 + b**2*cos(2*t)**2)**(3/2))
        dK_ds_list.append((-3*a*b*cos(t)*cos(2*t)/(a**2*cos(t)**2 + b**2*cos(2*t)**2)**(3/2) + (3*a**2*sin(t)*cos(t) + 6*b**2*sin(2*t)*cos(2*t))*(a*b*sin(t)*cos(2*t) - 2*a*b*sin(2*t)*cos(t))/(a**2*cos(t)**2 + b**2*cos(2*t)**2)**(5/2))/sqrt(a**2*cos(t)**2 + b**2*cos(2*t)**2))
    return points, headings, curvatures, dK_ds_list

#################################################################
# Path Helper Functions
#################################################################

def closest_point_idx(points, x, y):
    closest_idx = None
    closest_dist = None
    for i, p in enumerate(points):
        dist = sqrt((p[0] - x)**2 + (p[1] - y)**2)
        if closest_dist is None or dist < closest_dist:
            closest_dist = dist
            closest_idx = i
    return closest_idx

def closest_point_idx_local(points, x, y, prev_idx):
    """
    Performs a local search for the closest point.
    """
    closest_idx_forward = None
    closest_dist_forward = None
    # search forwards
    for i in range(prev_idx, prev_idx + len(points)):
        idx = i % len(points)
        dist = sqrt((points[idx][0] - x)**2 + (points[idx][1] - y)**2)
        if closest_dist_forward is None or dist < closest_dist_forward:
            closest_dist_forward = dist
            closest_idx_forward = idx
        else:
            break

    closest_idx_backward = None
    closest_dist_backward = None
    # search backwards
    for i in range(prev_idx, prev_idx - len(points), -1):
        idx = i % len(points)
        dist = sqrt((points[idx][0] - x)**2 + (points[idx][1] - y)**2)
        if closest_dist_backward is None or dist < closest_dist_backward:
            closest_dist_backward = dist
            closest_idx_backward = idx
        else:
            break
    
    assert closest_dist_forward is not None, 'No closest point found in the forward direction'
    assert closest_dist_backward is not None, 'No closest point found in the backward direction'
    if closest_dist_forward < closest_dist_backward:
        return closest_idx_forward
    else:
        return closest_idx_backward

def get_lookahead_idx(path_points, starting_idx, dist):
    remaining_dist = dist
    idx = starting_idx
    while remaining_dist > 0:
        if idx == len(path_points) - 1:
            idx = 0
        else:
            idx += 1
        remaining_dist -= sqrt((path_points[idx][0] - path_points[idx-1][0])**2 + (path_points[idx][1] - path_points[idx-1][1])**2)
    return idx

def walk_trajectory_by_durations(path_points, velocities, starting_idx, durations):
    assert all(d >= 0 for d in durations), "durations must be non-negative"
    assert len(durations) > 0, "Must have a duration"

    idx = starting_idx - 1
    remaining_segment_duration = 0.0
    indices = []
    for duration in durations:
        # first travel the untravelled portion of the current segment
        remaining_duration = max(0.0, duration - remaining_segment_duration)
        remaining_segment_duration -= duration - remaining_duration

        # at least one of two scenarios are possible here:
        # - we have used up the remaining segment duration, thus go into the loop to go onto the next segment
        # - we have not used up te remaining segment duration, so we stay on the current segment.
        # If both are true, we arbitrarily choose to record the existing segment index instead of the new one.
        # This is okay because we are already doing floating point calculations (minor errors are tolerable).
        while remaining_duration > 1e-15:
            # if this assertion is violated, then we have travelled more than the remaining duration, which is not possible
            assert -1e-15 < remaining_segment_duration < 1e-15, "remaining segment duration must be zero in the beginning of this loop"
            if remaining_segment_duration < 1e-15:
                # we have used up the remaining segment duration, progress onto the next segment
                idx = (idx+1)%len(path_points)

            segment_dist = sqrt((path_points[(idx+1)%len(path_points)][0] - path_points[idx][0])**2 + (path_points[(idx+1)%len(path_points)][1] - path_points[idx][1])**2)
            segment_duration = segment_dist / velocities[idx]
            
            remaining_segment_duration = max(0, segment_duration - remaining_duration)
            remaining_duration -= segment_duration - remaining_segment_duration

        indices.append(idx)
    return indices

#################################################################
# Controllers
#################################################################

class PIDController:
    def __init__(self, kp, ki, kd, dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt

        self.integral = 0
        self.prev_error = 0

    def step(self, error):
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error

        return self.kp * error + self.ki * self.integral + self.kd * derivative

#################################################################
# Research Functions
#################################################################

def optimize_l0(Phi: np.ndarray, Y: np.ndarray, eps: np.ndarray | float = 1e-15):
    """
    solves the l0 minimization problem
    Parameters:
        Phi: numpy.ndarray - tensor of size (N, q, n) that describes the evolution of the output over time.
            $N$ is the number of time steps, $q$ is the number of outputs, and $n$ is the number of states
        Y: numpy.ndarray - measured outputs, with input effects subtracted, size $(N, q)$
        eps: numpy.ndarray - noise tolerance for each output, size $(1,)$ or $(q,)$
    Returns:
        x0_hat: numpy.ndarray - estimated state, size $n$
        prob: cvxpy.Problem - the optimization problem
        metadata: dict - metadata about the optimization problem. Please see the code for the exact contents.
        solns: list - list of solutions for each possible set of corrupted sensors that was tried
    """

    N, q, n = Phi.shape
    assert Y.shape == (N, q)

    cvx_Y = Y.reshape((N*q,)) # groups of sensor measurements stacked vertically
    cvx_Phi = Phi.reshape((N*q, n)) # groups of transition+output matrices stacked vertically

    # Support scalar or vector eps
    if np.isscalar(eps):
        eps_final: np.ndarray = np.ones(q) * eps
    else:
        eps_final: np.ndarray = eps

    def optimize_case(corrupt_indices):
        x0_hat = cp.Variable(n)
        optimizer = cp.reshape(cvx_Y - np.matmul(cvx_Phi, x0_hat), (q, N))
        optimizer_final = cp.mixed_norm(optimizer, p=2, q=1)

        # Set toleance constraints to account for noise
        constraints = []
        for j in set(range(q)) - set(corrupt_indices):
            for k in range(N):
                constraints.append(optimizer[j][k] <= eps_final[j])
                constraints.append(optimizer[j][k] >= -eps_final[j])

        prob = cp.Problem(cp.Minimize(optimizer_final), constraints)

        start = time.time()
        prob.solve()
        end = time.time()
    
        return x0_hat, prob, {
            'corrupt_indices': corrupt_indices,
            'solve_time': end-start,
        }

    solns = [optimize_case(K) for K in powerset(range(q))]
    for x0_hat, prob, metadata in solns:
        if prob.status in ["optimal", "optimal_inaccurate"]:
            return (x0_hat, prob, metadata, solns)

    return (None, None, None, solns)

def get_state_evolution_tensor(As: list[np.ndarray]):
    """
    Given a list of $N-1$ state transition matrices, returns a tensor of size $(N,n,n)$ that describes the evolution of the state over time,
    where $n$ is the size of the state. The first element of the tensor is the identity matrix.
    Parameters:
        As: list[np.ndarray] - list of state transition matrices, size (n, n) each, where n is the number of states
    Returns:
        Evo: numpy.ndarray - tensor of size $(N, n, n)$ that describes the evolution of the state over time. When multiplied with the
            initial state, it produces the state at each time step.
    """
    assert len(As) > 0, "Must have at least one state transition matrix"

    N = len(As) + 1
    n = As[0].shape[0]
    Evo = np.zeros((N, n, n))
    Evo[0] = np.eye(n)
    for i in range(1, N):
        Evo[i] = np.matmul(As[i-1], Evo[i-1])
    return Evo

def get_output_evolution_tensor(Cs: list[np.ndarray], Evo: np.ndarray):
    """
    Given a list of $N$ output matrices and the state evolution tensor, returns a tensor of size
    $(N,q,n)$ that describes the evolution of the output over time, where $q$ is the number of
    outputs and $n$ is the number of states.

    Parameters:
        Cs: list[np.ndarray] - list of output matrices, size (q, n) each, where q is the number of outputs and n is the number of states
        Evo: numpy.ndarray - tensor of size $(N, n, n)$ that describes the evolution of the state over time. When multiplied with the
            initial state, it produces the state at each time step.
    Returns:
        Phi: numpy.ndarray - tensor of size $(N, q, n)$ that describes the evolution of the output over time.
    """
    assert len(Cs) > 0, "Must have at least one output matrix"
    assert Evo.shape[1] == Evo.shape[2], "State evolution matrices must be square"
    N = len(Cs)
    assert N == Evo.shape[0], "The number of output matrices must be equal to the size of the first dimension of the state evolution tensor"
    q = Cs[0].shape[0]
    n = Cs[0].shape[1]
    assert n == Evo.shape[1], "The number of states must match between the output matrices and the state evolution tensor"

    Phi = np.zeros((N, q, n))

    for i in range(N):
        Phi[i] = np.matmul(Cs[i], Evo[i])
    
    return Phi
        
