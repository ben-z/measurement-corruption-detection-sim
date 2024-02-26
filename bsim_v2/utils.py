import cvxpy as cp
import numpy as np
import json
import os
import time
import traceback
import fault_generators
import math
from collections import namedtuple
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from functools import wraps
from itertools import chain, combinations, repeat
from math import pi, sin, cos, atan2, sqrt
from numpy.typing import NDArray
from numpy.linalg import matrix_power
from typing import TypeVar, Iterable, Tuple, Optional, List, Any
from scipy.linalg import expm
import matplotlib.pyplot as plt
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm


MAX_POOL_SIZE = int(os.environ.get('MAX_POOL_SIZE', 240))

#################################################################
# General utility functions
#################################################################

def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def clamp(x, lower, upper):
    return np.maximum(lower, np.minimum(x, upper))


T = TypeVar('T')

def powerset(iterable: Iterable[T]) -> Iterable[Iterable[T]]:
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def get_unpack_fn(fn):
    """
    Returns a function that unpacks the first argument of the given function.
    """
    @wraps(fn)
    def unpack_fn(args):
        return fn(*args)

    return unpack_fn

def get_properties(obj):
    return [attr for attr in dir(obj) if isinstance(getattr(obj.__class__, attr, None), property)]

def format_floats(item, decimals=2):
    if isinstance(item, float):
        return f'{item:.{decimals}f}'
    elif isinstance(item, dict):
        return {key: format_floats(value, decimals) for key, value in item.items()}
    elif isinstance(item, (list, tuple)):
        return [format_floats(value, decimals) for value in item]
    elif isinstance(item, np.ndarray):
        vectorized_format = np.vectorize(lambda x: format_floats(x, decimals))
        return vectorized_format(item)
    else:
        return item

# Solves errors with JSON serialization such as "TypeError: Object of type 'int64' is not JSON serializable}"
# Derived from https://stackoverflow.com/a/57915246/4527337
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def calculate_segment_lengths(points: List[Tuple[float, float]]) -> List[float]:
    """Calculate the lengths of each segment given a list of (x, y) tuples."""
    return [
        math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        for (x1, y1), (x2, y2) in zip(points, np.roll(np.array(points), -1, axis=0))
    ]

#################################################################
# Models
#################################################################

# Kinematic bicycle model
def kinematic_bicycle_model(state, input, params):
    # Implements the kinematic bicycle model
    #
    # Parameters:
    #   state = [x, y, theta, v, delta]
    #   input = [a, delta_dot]
    #   params = {dt, l}
    # Returns:
    #   new_state = [x, y, theta, v, delta]
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

def kinematic_bicycle_model_linearize(theta, v, delta, dt, l):
    """
    Returns a linearized version of the kinematic bicycle model.
    Parameters:
        theta: float - the heading of the robot
        v: float - the velocity of the robot
        delta: float - the steering angle of the robot
        dt: float - the time step
        l: float - the length of the robot
    Returns:
        A: np.ndarray - the state transition matrix
        B: np.ndarray - the input matrix
    Model:
        State: [x, y, theta, v, delta]
        Input: [a, delta_dot]
    """
    A = np.array([
        [0, 0, -v*np.sin(theta), np.cos(theta), 0],
        [0, 0, v*np.cos(theta), np.sin(theta), 0],
        [0, 0, 0, np.tan(delta)/l, v*(1+np.tan(delta)**2)/l],
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

    # Calculate Ad and Bd at the same time: https://en.wikipedia.org/wiki/Discretization#cite_note-2
    ABd = expm(dt * np.block([
        [A, B],
        [np.zeros((B.shape[1], A.shape[0]+B.shape[1]))]
    ]))

    Ad = ABd[:A.shape[0], :A.shape[1]]
    Bd = ABd[:B.shape[0], A.shape[1]:]

    return Ad, Bd

def kinematic_bicycle_model_desired_state_at_idx(idx, path_points, path_headings, velocities):
    return np.array([path_points[idx][0], path_points[idx][1], path_headings[idx], velocities[idx], 0])

def kinematic_bicycle_model_normalize_output(output):
    return [
        output[0],
        output[1],
        wrap_to_pi(output[2]),
        output[3],
        wrap_to_pi(output[4]),
        wrap_to_pi(output[5]),
    ]

def calc_input_effects_on_output(As, Bs, Cs, inputs):
    # calculates the input effects on the output
    # Parameters:
    # As: numpy.ndarray[] - list of N-1 matrices of size (n,n)
    # Bs: numpy.ndarray[] - list of N-1 matrices of size (n,p)
    # Cs: numpy.ndarray[] - list of N matrices of size (q,n)
    # inputs: numpy.ndarray[] - list of N-1 input vectors of size p
    # Returns:
    # output_effects: numpy.ndarray[] - the effects of the inputs on the output, a list of N matrices of size (q,)

    n = As[0].shape[0]
    N = len(Cs)
    assert len(As) == N-1, "As must have length N-1"
    assert len(Bs) == N-1, "Bs must have length N-1"
    assert len(inputs) == N-1, "inputs must have length N-1"

    # Algorithm (for LTV systems):
    # zeros
    # B[0]u[0]
    # B[1]u[1] + A[0](B[0]u[0])
    # B[2]u[2] + A[1](B[1]u[1] + A[0](B[0]u[0]))
    # ...
    # Then pass everything through C
    state_effects = [
        np.zeros((n,)),
        np.matmul(Bs[0], inputs[0])
    ]
    for i in range(1, N-1):
        state_effects.append(np.matmul(Bs[i], inputs[i]) + np.matmul(As[i-1], state_effects[-1]))

    output_effects = [Cs[i]@state_effects[i] for i in range(N)]

    return output_effects

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
    """
    Walks a trajectory by the given durations. Returns the indices of the path points that were travelled to.
    Parameters:
        path_points: list[tuple[float, float]] - list of path points denoting x and y coordinates over time
        velocities: list[float] - list of velocities at each path point
        starting_idx: int - the index of the starting path point
        durations: list[float] - list of durations to travel for each segment. This is non-cumulative.
    Returns:
        indices: list[int] - list of indices of path points that were travelled to
    """
    assert len(path_points) == len(velocities), f"path_points and velocities must have the same length. {len(path_points)=}, {len(velocities)=}"
    assert starting_idx >= 0 and starting_idx < len(path_points), "starting_idx must be a valid index"
    assert len(durations) > 0, "Must have a duration"
    assert all(d >= 0 for d in durations), "durations must be non-negative"

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

# Stripped down cvxpy.Problem that is serializable
MyCvxpyProblem = namedtuple('MyCvxpyProblem', ['status', 'solver_stats', 'compilation_time', 'solve_time', 'value'])
MyOptimizationCaseResult = Tuple[NDArray[np.float64] | None, MyCvxpyProblem, dict]
MyOptimizerRes = Tuple[Optional[MyOptimizationCaseResult], List[MyOptimizationCaseResult], dict]

def optimize_l0_v2(Phi: np.ndarray, Y: np.ndarray, eps: NDArray[np.float64] | float = 1e-15, S_list: Optional[Iterable[Iterable[int]]] = None, solver_args: dict = {}, **_kwargs) -> MyOptimizerRes:
    r"""
    solves the l0 minimization problem. i.e. attempt to explain the output $Y$ using the model $\Phi$ (`Phi`) and return
    the most-likely initial state $\hat{x}_0$ (`x0_hat`) and the corrupted sensors (`K`).
    Parameters:
        Phi: numpy.ndarray - tensor of size (N, q, n) that describes the evolution of the output over time.
            $N$ is the number of time steps, $q$ is the number of outputs, and $n$ is the number of states
        Y: numpy.ndarray - measured outputs, with input effects subtracted, size $(N, q)$
        eps: numpy.ndarray - noise tolerance for each output, size $(1,)$ or $(q,)$
        S_list: Optional[Iterable[Iterable[int]]] - list of sensor combinations to try. If None, then all possible sensor combinations are tried.
            This is useful when you know that some sensors are not corrupted.
    Returns:
        x0_hat: numpy.ndarray - estimated state, size $n$
        prob: cvxpy.Problem - the optimization problem
        metadata: dict - metadata about the optimization problem. Please see the code for the exact contents.
        solns: list - list of solutions for each possible set of corrupted sensors that was tried
    """
    metadata = {}

    start = time.perf_counter()
    N, q, n = Phi.shape
    assert Y.shape == (N, q)

    cvx_Y = Y.reshape((N*q,)) # groups of sensor measurements stacked vertically
    cvx_Phi = Phi.reshape((N*q, n)) # groups of transition+output matrices stacked vertically

    # Support scalar or vector eps
    if np.isscalar(eps):
        eps_final: NDArray[np.float64] = np.ones(q) * eps # type: ignore
    else:
        eps_final: NDArray[np.float64] = eps # type: ignore

    def optimize_case(S):
        """
        Solves the l0 minimization problem for a given set of uncorrupted sensors $S$.
        """
        # K is the sensors that can be corrupted (i.e. the sensors that are not in S)
        K = list(set(range(q)) - set(S))

        x0_hat = cp.Variable(n)
        optimizer = cp.reshape(cvx_Y - np.matmul(cvx_Phi, x0_hat), (q, N)) # type: ignore
        optimizer_final = cp.mixed_norm(optimizer, p=2, q=1)

        # Set toleance constraints to account for noise
        constraints = []
        for j in S:
            for k in range(N):
                constraints.append(optimizer[j][k] <= eps_final[j])
                constraints.append(optimizer[j][k] >= -eps_final[j])

        prob = cp.Problem(cp.Minimize(optimizer_final), constraints)

        start = time.time()
        prob.solve(**solver_args)
        end = time.time()
    
        return x0_hat.value, MyCvxpyProblem(
            status=prob.status,
            solver_stats=prob.solver_stats,
            compilation_time=prob.compilation_time,
            solve_time=prob._solve_time,
            value=prob.value,
        ), {
            'K': K,
            'S': S,
            'solve_time': end-start,
        }

    # sort the list of sensor combinations by size, largest to smallest
    # this is because the optimization algorithm needs to minimize the number of corrupt sensors
    S_list = sorted(S_list or powerset(range(q)), key=lambda S: len(list(S)), reverse=True)
    end = time.perf_counter()
    metadata['setup_time'] = end-start

    start = time.perf_counter()
    solns = [optimize_case(S) for S in S_list]
    end = time.perf_counter()
    metadata['solve_time'] = end-start
    for soln in solns:
        _x0_hat, prob, _metadata = soln
        if prob.status in ["optimal", "optimal_inaccurate"]:
            return soln, solns, metadata

    return None, solns, metadata

class Optimizer:
    def __init__(self, N: int, q: int, n: int, solver: str = cp.CLARABEL):
        self.N = N
        self.q = q
        self.n = n
        self.eps_param = cp.Parameter((q,))
        self.cvx_Y_param = cp.Parameter((self.N*self.q,))
        self.cvx_Phi_param = cp.Parameter((self.N*self.q, self.n))
        self.solver = solver

        self.x0_hat = cp.Variable(self.n)
        optimizer = cp.reshape(self.cvx_Y_param - self.cvx_Phi_param @ self.x0_hat, (self.q, self.N))
        optimizer_final = cp.mixed_norm(optimizer, p=2, q=1)

        self.can_corrupt = cp.Parameter(self.q, boolean=True)
        self.can_corrupt.value = np.ones(self.q)
        slack = cp.Variable(self.q)
        constraints = []
        for j in range(self.q):
            for k in range(self.N):
                constraints.append(cp.abs(optimizer[j][k]) <= self.eps_param[j] + cp.multiply(self.can_corrupt[j], slack[j]))
        self.prob = cp.Problem(cp.Minimize(optimizer_final), constraints)
        # Warm up problem data cache. This should make compilation much faster see (prob.compilation_time)
        self.prob.get_problem_data(self.solver)
    
    def optimize_l0_v4(
        self,
        Phi: np.ndarray,
        Y: np.ndarray,
        eps: NDArray[np.float64] | float = 1e-15,
        S_list: Optional[Iterable[Iterable[int]]] = None,
        solver_args: dict = {},
        early_exit: bool = True,
    ) -> MyOptimizerRes:
        r"""
        solves the l0 minimization problem. i.e. attempt to explain the output $Y$ using the model $\Phi$ (`Phi`) and return
        the most-likely initial state $\hat{x}_0$ (`x0_hat`) and the corrupted sensors (`K`).
        Parameters:
            Phi: numpy.ndarray - tensor of size (N, q, n) that describes the evolution of the output over time.
                $N$ is the number of time steps, $q$ is the number of outputs, and $n$ is the number of states
            Y: numpy.ndarray - measured outputs, with input effects subtracted, size $(N, q)$
            eps: numpy.ndarray - noise tolerance for each output, size $(1,)$ or $(q,)$
            S_list: Optional[Iterable[Iterable[int]]] - list of sensor combinations to try. If None, then all possible sensor combinations are tried.
                This is useful when you know that some sensors are not corrupted.
            solver_args: dict - arguments to pass to the cp.Problem().solve function
        Returns:
            x0_hat: numpy.ndarray - estimated state, size $n$
            prob: MyCvxpyProblem- the optimization problem
            metadata: dict - metadata about the optimization problem. Please see the code for the exact contents.
            solns: list - list of solutions for each possible set of corrupted sensors that was tried
        """
        metadata = {}
        start = time.perf_counter()

        N, q, n = Phi.shape
        assert Y.shape == (N, q), f"{Y.shape=} must be equal to {(N, q)=}"
        assert N == self.N, f"N must be equal to {self.N=}"
        assert q == self.q, f"q must be equal to {self.q=}"
        assert n == self.n, f"n must be equal to {self.n=}"

        cvx_Y = Y.reshape((N*q,)) # groups of sensor measurements stacked vertically
        cvx_Phi = Phi.reshape((N*q, n)) # groups of transition+output matrices stacked vertically

        # sort the list of sensor combinations by size, largest to smallest
        # this is because the optimization algorithm needs to minimize the number of corrupt sensors
        S_list = sorted(S_list or powerset(range(q)), key=lambda S: len(list(S)), reverse=True)

        # Support scalar or vector eps
        self.eps_param.value = np.broadcast_to(eps, (q,))
        self.cvx_Y_param.value = cvx_Y
        self.cvx_Phi_param.value = cvx_Phi

        end = time.perf_counter()
        metadata['setup_time'] = end-start

        start = time.perf_counter()
        map_args = [S_list, repeat(q), repeat(self.prob), repeat(self.x0_hat), repeat(self.can_corrupt), repeat({'solver': self.solver, **solver_args})]
        soln_generator = map(optimize_l0_case, *map_args)
        # with Pool(min(MAX_POOL_SIZE, os.cpu_count() or 1)) as pool:
        #   soln_generator = pool.starmap(optimize_l0_case, zip(*map_args))

        metadata['solutions_with_errors'] = []

        ret = None
        solns = []
        try:
            for s in soln_generator:
                s_x0_hat, s_prob, s_metadata = s
                solns.append(s)
                if s_metadata.get('solver_error'):
                    metadata['solutions_with_errors'].append(s_metadata)
                if ret is None and s_prob.status in ["optimal", "optimal_inaccurate"]:
                    ret = (s_x0_hat, s_prob, s_metadata)
                    if early_exit:
                        break
        finally:
            end = time.perf_counter()
            metadata['solve_time'] = end-start

        return ret, solns, metadata

def optimize_l0_case(
    S: Iterable[int],
    q: int,
    prob: cp.Problem,
    x0_hat: cp.Variable,
    can_corrupt: cp.Parameter,
    solver_args: dict = {},
) -> MyOptimizationCaseResult:
    r"""
    Solves the l0 minimization problem for a given set of uncorrupted sensors $S$.
    Parameters:
        S: Iterable[int] - the set of uncorrupted sensors
        q: int - the number of outputs
        prob: cp.Problem - the optimization problem
        x0_hat: cp.Variable - the variable to optimize
        can_corrupt: cp.Parameter - a parameter that indicates which sensors can be corrupted
        solver_args: dict - arguments to pass to the solver
    Returns:
        x0_hat: numpy.ndarray - estimated state, size $n$
        prob: MyCvxpyProblem - the optimization problem
        metadata: dict - metadata about the optimization problem. Please see the code for the exact contents.
    """
    additional_metadata = {}

    # K is the sensors that can be corrupted (i.e. the sensors that are not in S)
    K = list(set(range(q)) - set(S))

    can_corrupt.value = np.ones(q)
    for j in S:
        can_corrupt.value[j] = False

    start = time.perf_counter()
    try:
        prob.solve(**solver_args)
    except cp.SolverError as e:
        print(f"Solver error when solving for {S=}: {e}")
        additional_metadata["solver_error"] = str(e)
    except Exception as e:
        print(f"Unknown error when solving for {S=}: {e}")
        additional_metadata["solver_error"] = str(e)
    end = time.perf_counter()

    return x0_hat.value, MyCvxpyProblem(
        status=prob.status,
        solver_stats=prob.solver_stats,
        compilation_time=prob.compilation_time,
        solve_time=prob._solve_time,
        value=prob.value,
    ), {
        'K': K,
        'S': S,
        'solve_time': end-start,
        **additional_metadata,
    }

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

def get_solver_setup(output_hist, input_hist, closest_idx_hist, path_points, path_headings, velocities, Cs, dt, l, model_at_idx, desired_output_fn, normalize_output):
    """
    Returns the precomputed data needed to solve for corrupted sensors.
    Parameters:
        output_hist: list[float] - a list of N outputs
        input_hist: list[float] - a list of N-1 inputs
        closest_idx_hist: list[int] - a list of N-1 closest indices
        path_points: list[tuple[float,float]] - a list of points on the path
        velocities: list[float] - a list of velocities at each point on the path
        dt: float - the time step
        l: float - the distance between the front and rear axles
    """
    N = len(output_hist)
    assert len(input_hist) == N-1, 'input_hist must be one shorter than output_hist'
    assert len(closest_idx_hist) == N-1, 'closest_idx_hist must be one shorter than output_hist'
    assert len(Cs) == N, 'Cs must be N long'

    x0_hat_closest_idx = closest_idx_hist[0]

    # Use walk_trajectory_by_durations to get the desired path indices at each time step
    # TODO: exp:use-real-indices - use closest_idx_hist instead of desired_path_indices
    desired_path_indices = [x0_hat_closest_idx] + walk_trajectory_by_durations(path_points, velocities, x0_hat_closest_idx, [dt]*(N-1))
    # desired_path_indices = closest_idx_hist + [closest_idx_hist[-1]]

    # Generate the As and Bs for each time step
    # model_fn
    # models = [kinematic_bicycle_model_linearize(path_headings[idx], velocities[idx], 0, dt, l) for idx in desired_path_indices]
    models = [model_at_idx(idx) for idx in desired_path_indices]
    As = list(m[0] for m in models[:N-1])
    Bs = list(m[1] for m in models[:N-1])

    # List of desired outputs at the linearization points
    # desired_output_at_idx(idx)
    # desired_trajectory = [Cs[i] @ np.array([path_points[idx][0], path_points[idx][1], path_headings[idx], velocities[idx], 0]) for i, idx in enumerate(desired_path_indices)]
    desired_trajectory = [desired_output_fn(i, idx) for i, idx in enumerate(desired_path_indices)]

    # Calculate the effects on outputs due to inputs, then subtract them from the outputs.
    # Also subtract the desired outputs to get deviations.
    input_effects = calc_input_effects_on_output(As, Bs, Cs, input_hist)
    output_hist_no_input_effects = [output - input_effect - desired_output for output, input_effect, desired_output in zip(output_hist, input_effects, desired_trajectory)]
    # normalize angular measurements
    # normalize_output
    for i in range(len(output_hist_no_input_effects)):
        output_hist_no_input_effects[i] = normalize_output(output_hist_no_input_effects[i])
        # o[2] = wrap_to_pi(o[2])
        # o[4] = wrap_to_pi(o[4])
        # o[5] = wrap_to_pi(o[5])

    # Solve for corrupted sensors
    Y = np.array(output_hist_no_input_effects)
    Phi = get_output_evolution_tensor(Cs, get_state_evolution_tensor(As))

    return {
        'Y': Y,
        'Phi': Phi,
        'output_hist_no_input_effects': output_hist_no_input_effects,
    }

def get_s_sparse_observability(Cs, As, early_exit=False):
    """
    Returns the s-sparse observability for the given system.

    Parameters:
        Cs: list[np.ndarray] - list of output matrices, size $(q, n)$ each, where q is the number of outputs and n is the number of states
        As: list[np.ndarray] - list of state transition matrices, size $(n, n)$ each, where n is the number of states
        early_exit: bool - whether or not to stop after finding $s$.
    Requires:
        len(Cs) == len(As) + 1
    Returns:
        s: int - the s-sparse observability of the system
        cases: list[(np.ndarray, bool)] - list of all possible sensor combinations and whether or not they are observable.
    """
    assert len(Cs) == len(As) + 1, "The number of output matrices must be one more than the number of state transition matrices"
    N = len(Cs)
    q = Cs[0].shape[0]
    n = Cs[0].shape[1]

    if len(As) == 0:
        Evo = np.eye(n)
    else:
        Evo = get_state_evolution_tensor(As)

    Phi = get_output_evolution_tensor(Cs, Evo)

    # We want to find the largest s such that removing s sensors still leaves the system observable.
    s = None
    cases = []
    # K is the missing sensors
    for K in map(list, powerset(range(q))):
        # S is the sensors that are not missing
        S = list(set(range(q)) - set(K))
        if len(S) == 0:
            # if we are missing all sensors, then the system is not observable
            cases.append((S, False))
            continue

        Phi_S = Phi[:, S, :]

        O_S = Phi_S.reshape((N * len(S), n))
        
        O_S_rank = np.linalg.matrix_rank(O_S)

        cases.append((S, O_S_rank == n))

        if s is None and O_S_rank < n:
            # found the first non-full-rank entry
            s = len(K) - 1

            if early_exit:
                break

    return s, cases

def estimate_state(
    kf,
    output_hist,
    input_hist,
    closest_idx_hist,
    path_points,
    path_headings,
    velocities,
    Cs,
    dt,
    l,
    N,
    enable_fault_tolerance,
    optimizer,
    noise_std,
    model_at_idx,
    desired_output_fn,
    normalize_output,
    enable_estimator = True,
) -> Tuple[NDArray[np.float64] | None, MyOptimizerRes | None, dict]:
    """
    Estimates the state of the vehicle.
    Parameters:
        kf: Optional[filterpy.kalman.UnscentedKalmanFilter] - the filter to use. If None, simply average the sensors.
        output_hist: list[float] - a list of N outputs
        input_hist: list[float] - a list of N-1 inputs
        closest_idx_hist: list[float] - a list of N-1 indices of the path where the vehicle was closest to at each time step
        path_points: list[tuple[float,float]] - a list of points on the path
        velocities: list[float] - a list of velocities at each point on the path
        dt: float - the time step
    Returns:
        (x_hat, y_hat, theta_hat, v_hat, delta_hat): tuple[float,float,float,float,float] - the estimated state
        optimizer_res: Optional[Optional[MyOptimizationCaseResult], List[MyOptimizationCaseResult]] - the result of the optimizer
        metadata: dict - metadata about the estimation process
    """
    metadata = {}
    start = start_e2e = time.perf_counter()
    if kf is not None:
        if len(input_hist) == 0:
            kf.predict(u=np.array([0,0]))
        else:
            kf.predict(u=input_hist[-1])

        # print(kf.x)
        # print(kf.P)

    # This is the latest output
    output = output_hist[-1]
    if enable_estimator:
        assert len(output) == 6, 'The estimator currently only works with output (x,y,theta,v,delta1,delta2)'

    if not enable_estimator:
        # skip the estimator
        x_hat = None
    elif kf is not None:
        kf.update(output)

        x_hat = kf.x
    else:
        # State estimator
        x_hat = np.zeros(5)
        x_hat[0] = output[0]
        x_hat[1] = output[1]
        x_hat[2] = output[2]
        x_hat[3] = output[3]
        x_hat[4] = np.mean([output[4], output[5]])
    end = time.perf_counter()
    metadata['filter_time'] = end - start

    if not enable_fault_tolerance:
        return x_hat, None, metadata

    ###########################################################################
    # Fault detection
    ###########################################################################
    if len(output_hist) < N:
        print(f"Insufficient data to solve for corrupt sensors. Need {N} outputs, only have {len(output_hist)}")
        return x_hat, None, metadata

    assert len(input_hist) == N-1, 'input_hist must be one shorter than output_hist'
    assert len(closest_idx_hist) == N-1, 'closest_idx_hist must be one shorter than output_hist'
    assert len(output_hist) == N, 'output_hist must be N long'

    setup = get_solver_setup(
        output_hist,
        input_hist,
        closest_idx_hist,
        path_points,
        path_headings,
        velocities,
        Cs,
        dt,
        l,
        model_at_idx,
        desired_output_fn,
        normalize_output,
    )

    # eps = np.array([0.1]*2+[1e-2]+[1e-3]*3)
    # eps = np.array([1.0]*2+[0.3]+[1.5]+[0.05]*2)
    eps = noise_std * 3 # 3 standard deviations captures 99.7% of the data

    start = time.perf_counter()
    optimizer_res: MyOptimizerRes = optimizer.optimize_l0_v4(setup['Phi'], setup['Y'], eps=eps, solver_args={'solver': cp.CLARABEL})
    # optimizer_res = optimize_l0_v2(setup['Phi'], setup['Y'], eps=eps, solver_args={'solver': cp.CLARABEL})
    end = end_e2e = time.perf_counter()
    metadata['optimizer_time'] = end - start
    metadata['total_time'] = end_e2e - start_e2e
    # print(f"Optimizer time: {metadata['optimizer_time']:.3f}s, Total time: {metadata['total_time']:.3f}s")

    return x_hat, optimizer_res, metadata

def estimate_state_unpack(args):
    return estimate_state(*args)

def run_simulation(x0, C, noise_std, num_steps, N, path_points, path_headings, path_curvatures, path_dcurvatures, velocity_profile, optimizer, model_params, attack_generator, enable_fault_tolerance):
    """
    Runs the simulation.
    """
    # Functions specific to the model used
    def subtract_states(x1, x2):
        return np.array([
            x1[0] - x2[0],
            x1[1] - x2[1],
            wrap_to_pi(x1[2] - x2[2]),
            x1[3] - x2[3],
            wrap_to_pi(x1[4] - x2[4])
        ])

    def subtract_outputs(y1, y2):
        return np.array([
            y1[0] - y2[0],
            y1[1] - y2[1],
            wrap_to_pi(y1[2] - y2[2]),
            y1[3] - y2[3],
            wrap_to_pi(y1[4] - y2[4]),
            wrap_to_pi(y1[5] - y2[5])
        ])

    # TODO: verify the implementations of state_mean and z_mean
    def state_mean(sigmas, Wm):
        x = np.zeros(5)
        sum_sin = np.sum(np.dot(np.sin(sigmas[:,2]), Wm))
        sum_cos = np.sum(np.dot(np.cos(sigmas[:,2]), Wm))
        x[0] = np.dot(sigmas[:,0], Wm)
        x[1] = np.dot(sigmas[:,1], Wm)
        x[2] = atan2(sum_sin, sum_cos)
        x[3] = np.dot(sigmas[:,3], Wm)
        x[4] = np.dot(sigmas[:,4], Wm)
        return x

    def z_mean(sigmas, Wm):
        z = np.zeros(6)
        sum_sin = np.sum(np.dot(np.sin(sigmas[:,2]), Wm))
        sum_cos = np.sum(np.dot(np.cos(sigmas[:,2]), Wm))
        z[0] = np.dot(sigmas[:,0], Wm)
        z[1] = np.dot(sigmas[:,1], Wm)
        z[2] = atan2(sum_sin, sum_cos)
        z[3] = np.dot(sigmas[:,3], Wm)
        z[4] = np.dot(sigmas[:,4], Wm)
        z[5] = np.dot(sigmas[:,5], Wm)
        return z

    def get_lookahead_distance(v):
        return max(v * 0.5, 0.5)

    state = x0
    # TODO: tune the sigma points
    ukf_sigma_points = MerweScaledSigmaPoints(n=C.shape[1], alpha=0.3, beta=2, kappa=3-C.shape[1], subtract=subtract_states)
    ukf = UnscentedKalmanFilter(
        dim_x=C.shape[1],
        dim_z=C.shape[0],
        dt=model_params['dt'],
        fx=lambda x, _dt, u: kinematic_bicycle_model(x, u, model_params),
        hx=lambda x: C @ x,
        points=ukf_sigma_points,
        residual_x=subtract_states,
        residual_z=subtract_outputs,
        x_mean_fn=state_mean,
        z_mean_fn=z_mean,
    )
    ukf.x = x0
    ukf.P = np.diag([1,1,0.3,0.5,0.1]) # initial state covariance
    ukf.R = np.diag(noise_std**2) # measurement noise
    ukf.Q = np.diag([0.1,0.1,0.01,0.1,0.001]) # process noise
    state_hist = []
    output_hist = []
    estimate_hist = []
    u_hist = []
    closest_idx_hist = []
    ukf_P_hist = []
    a_controller = PIDController(2, 0, 0, model_params['dt'])
    delta_dot_controller = PIDController(5, 0, 0, model_params['dt'])
    prev_closest_idx = None
    for i in range(num_steps):
        state_hist.append(state)

        # measurement
        output = C @ state
        # Add noise
        output += np.random.normal(0, noise_std**2)

        # Add attack
        output = attack_generator(i * model_params['dt'], output)

        output_hist.append(output)

        # fault-tolerant estimator
        estimator_res, optimizer_res, _ = estimate_state(
            ukf,
            output_hist[-N:],
            u_hist[-(N-1):],
            closest_idx_hist[-(N-1):],
            path_points,
            path_headings,
            velocity_profile,
            [C]*N,
            dt=model_params['dt'],
            l=model_params['l'],
            N=N,
            enable_fault_tolerance=enable_fault_tolerance,
            optimizer=optimizer,
            noise_std=noise_std,
            model_at_idx=lambda idx: kinematic_bicycle_model_linearize(path_headings[idx], velocity_profile[idx], 0, model_params['dt'], model_params['l']),
            desired_output_fn=lambda i, idx: C @ kinematic_bicycle_model_desired_state_at_idx(idx, path_points, path_headings, velocity_profile),
            normalize_output=kinematic_bicycle_model_normalize_output,
        )
        # if optimizer_res is None:
        #     print(f"k={i}: Optimizer not run")
        # elif optimizer_res[0] is None:
        #     print(f"k={i}: Optimizer failed")
        # else:
        #     print(f"k={i}: {optimizer_res[0][2]['K']} corrupted")

        assert estimator_res is not None, 'Estimator returned None! This should not happen.'
        x_hat, y_hat, theta_hat, v_hat, delta_hat = estimator_res

        estimate_hist.append([x_hat, y_hat, theta_hat, v_hat, delta_hat])
        ukf_P_hist.append(ukf.P.copy())

        if prev_closest_idx is None:
            closest_idx = closest_point_idx(path_points, x_hat, y_hat)
        else:
            closest_idx = closest_point_idx_local(path_points, x_hat, y_hat, prev_closest_idx)
        assert closest_idx is not None, 'No closest point found'
        closest_idx_hist.append(closest_idx)
        prev_closest_idx = closest_idx

        lookahead_distance = get_lookahead_distance(v_hat)
        target_idx = get_lookahead_idx(path_points, closest_idx, lookahead_distance)

        target_point = path_points[target_idx]
        target_heading = path_headings[target_idx]
        target_curvature = path_curvatures[target_idx]
        target_dcurvature = path_dcurvatures[target_idx]
        target_velocity = velocity_profile[target_idx]

        # Pure pursuit controller
        dist_to_target = sqrt((target_point[0] - x_hat)**2 + (target_point[1] - y_hat)**2)
        angle_to_target = atan2(target_point[1] - y_hat, target_point[0] - x_hat) - theta_hat
        target_delta = atan2(2*model_params['l']*sin(angle_to_target), dist_to_target)

        # Compute the control inputs (with saturation)
        a = clamp(a_controller.step(target_velocity - v_hat), -model_params["max_linear_acceleration"], model_params["max_linear_acceleration"])
        delta_dot = clamp(delta_dot_controller.step(wrap_to_pi(target_delta - delta_hat)), -model_params["max_steering_rate"], model_params["max_steering_rate"])

        # Simulate the bicycle
        state = kinematic_bicycle_model(state, [a, delta_dot], model_params)
        u_hist.append([a, delta_dot])
    t_hist = [i * model_params['dt'] for i in range(num_steps)]

    return t_hist, state_hist, output_hist, estimate_hist, u_hist, closest_idx_hist, ukf_P_hist

# Plot simulation data
def plot_quad(t_hist, state_hist, output_hist, estimate_hist, u_hist, closest_idx_hist, ukf_P_hist, path_points, path_headings, velocity_profile, model_params):
    from filterpy.stats import plot_covariance

    num_steps = len(t_hist)

    EGO_COLOR = 'tab:orange'
    EGO_ESTIMATE_COLOR = 'tab:red'
    EGO_ACTUATION_COLOR = 'tab:green'
    TARGET_COLOR = 'tab:blue'
    TITLE = "Simulation Data"
    FIGSIZE_MULTIPLIER = 1.5
    fig = plt.figure(figsize=(6.4 * FIGSIZE_MULTIPLIER, 4.8 * FIGSIZE_MULTIPLIER), constrained_layout=True)
    suptitle = fig.suptitle(TITLE)
    subfigs = fig.subfigures(2, 2)
    COV_INTERVAL_S = 5 # number of seconds between covariance ellipses
    uncertainty_std = 1 # number of standard deviations to plot for uncertainty

    # BEV plot
    ax_bev = subfigs[0][0].add_subplot(111)
    ax_bev.plot([p[0] for p in path_points], [p[1] for p in path_points], '.', label='path', markersize=0.1, color=TARGET_COLOR)
    ego_position = ax_bev.plot([p[0] for p in state_hist], [p[1] for p in state_hist], '.', markersize=0.1, color=EGO_COLOR, label='ego')[0]
    ego_position_estimate = ax_bev.plot([p[0] for p in estimate_hist], [p[1] for p in estimate_hist], '.', markersize=0.1, color=EGO_ESTIMATE_COLOR, label='ego estimate')[0]
    # Velocity plot
    ax_velocity = subfigs[0][1].add_subplot(111)
    ax_velocity.plot(t_hist, [velocity_profile[idx] for idx in closest_idx_hist], label=r"$v_d$", color=TARGET_COLOR) # target velocity
    ax_velocity.plot(t_hist, [p[3] for p in state_hist], label=r"$v$", color=EGO_COLOR) # velocity
    ax_velocity.plot(t_hist, [p[3] for p in estimate_hist], label=r"$\hat{v}$", color=EGO_ESTIMATE_COLOR) # velocity estimate
    ax_velocity.fill_between(t_hist, [p[3] - uncertainty_std*np.sqrt(ukf_P[3,3]) for p, ukf_P in zip(estimate_hist, ukf_P_hist)], [p[3] + uncertainty_std*np.sqrt(ukf_P[3,3]) for p, ukf_P in zip(estimate_hist, ukf_P_hist)], alpha=0.2, color=EGO_ESTIMATE_COLOR)
    # Heading&Steering plot
    ax_heading_steering = subfigs[1][0].subplots(2, 1, sharex=True)
    ax_heading = ax_heading_steering[0]
    ax_heading.plot(t_hist, np.unwrap([path_headings[idx] for idx in closest_idx_hist]), label=r"$\theta_d$", color=TARGET_COLOR) # target heading
    ax_heading.plot(t_hist, np.unwrap([p[2] for p in state_hist]), label=r"$\theta$", color=EGO_COLOR) # heading
    ax_heading.plot(t_hist, np.unwrap([p[2] for p in estimate_hist]), label=r"$\hat{\theta}$", color=EGO_ESTIMATE_COLOR) # heading estimate
    ax_heading.fill_between(t_hist, np.unwrap([p[2] - uncertainty_std*np.sqrt(ukf_P[2,2]) for p, ukf_P in zip(estimate_hist, ukf_P_hist)]), np.unwrap([p[2] + uncertainty_std*np.sqrt(ukf_P[2,2]) for p, ukf_P in zip(estimate_hist, ukf_P_hist)]), alpha=0.2, color=EGO_ESTIMATE_COLOR)
    ax_steering = ax_heading_steering[1]
    ax_steering.plot(t_hist, [p[4] for p in state_hist], label=r"$\delta$", color=EGO_COLOR) # steering
    ax_steering.plot(t_hist, [p[4] for p in estimate_hist], label=r"$\hat{\delta}$", color=EGO_ESTIMATE_COLOR) # steering estimate
    ax_steering.fill_between(t_hist, [p[4] - uncertainty_std*np.sqrt(ukf_P[4,4]) for p, ukf_P in zip(estimate_hist, ukf_P_hist)], [p[4] + uncertainty_std*np.sqrt(ukf_P[4,4]) for p, ukf_P in zip(estimate_hist, ukf_P_hist)], alpha=0.2, color=EGO_ESTIMATE_COLOR)
    # Control signals plot
    axes_control = subfigs[1][1].subplots(2, 1, sharex=True)
    axes_control[0].plot(t_hist, [u[0] for u in u_hist], label=r"$a$", color=EGO_ACTUATION_COLOR) # a
    axes_control[1].plot(t_hist, [u[1] for u in u_hist], label=r"$\dot{\delta}$", color=EGO_ACTUATION_COLOR) # delta_dot

    # # Plot covariance ellipses
    # plt.sca(ax_bev)
    # for est, ukf_P in zip(estimate_hist[::int(COV_INTERVAL_S/model_params['dt'])], ukf_P_hist[::int(COV_INTERVAL_S/model_params['dt'])]):
    #     plot_covariance(est[:2], std=100, cov=ukf_P[0:2, 0:2], facecolor='none', edgecolor=EGO_ESTIMATE_COLOR)

    # Finalize plots
    ax_bev.axis('equal')
    ax_bev.set_title('BEV')
    ax_bev.legend(markerscale=50)
    ax_velocity.set_title("Velocity over time")
    ax_velocity.set_xlabel(r'Time ($s$)')
    ax_velocity.set_ylabel(r'Velocity ($m/s$)')
    ax_velocity.legend()
    ax_heading.set_title("Heading and Steering over time")
    ax_heading.set_ylabel(r'Heading ($rad$)')
    ax_heading.legend()
    ax_steering.set_xlabel(r'Time ($s$)')
    ax_steering.set_ylabel(r'Steering ($rad$)')
    ax_steering.legend()
    axes_control[0].set_title("Control signals over time")
    axes_control[0].set_ylabel(r'Acceleration ($m/s^2$)')
    axes_control[0].legend()
    axes_control[1].set_xlabel(r'Time ($s$)')
    axes_control[1].set_ylabel(r'Steering rate ($rad/s$)')
    axes_control[1].legend()

    def generate_gif():
        from matplotlib.animation import PillowWriter, FuncAnimation
        ANIM_TITLE_FORMAT = 'Simulation Playback (Current time: ${:.2f}$ s)'

        # add time cursors
        time_cursors = []
        time_cursors.append(ax_velocity.axvline(0, color='k'))
        time_cursors.append(ax_heading.axvline(0, color='k'))
        time_cursors.append(axes_control[0].axvline(0, color='k'))
        time_cursors.append(axes_control[1].axvline(0, color='k'))
        def animate(i):
            # ================= Update title =================
            suptitle.set_text(ANIM_TITLE_FORMAT.format(i * model_params['dt']))

            # =============== Plot ego position ===============
            # For a moving position, use this:
            #ego_position.set_data(state_hist[i][0], state_hist[i][1])
            # To plot the entire trajectory, use this:
            ego_position.set_data([p[0] for p in state_hist[:i+1]], [p[1] for p in state_hist[:i+1]])
            ego_position_estimate.set_data([p[0] for p in estimate_hist[:i+1]], [p[1] for p in estimate_hist[:i+1]])

            # =============== Plot time cursors ===============
            for time_cursor in time_cursors:
                time_cursor.set_xdata([i * model_params['dt']])

            return suptitle, ego_position, *time_cursors

        anim_interval_ms = 2000 # the step size of the animation in milliseconds
        anim_frames: list[Any] = list(range(0, num_steps, int(anim_interval_ms / 1000 / model_params['dt'])))
        anim = FuncAnimation(fig, animate, frames=anim_frames, interval=anim_interval_ms)

        start = time.perf_counter()
        # anim.save('zero_state.gif', writer=PillowWriter(fps=1000/anim_interval_ms))

        from IPython.core.display import HTML
        html = HTML(anim.to_jshtml())
        print(f"Animation generation took {time.perf_counter() - start:.2f} seconds")
        plt.close(fig)

        return html
    
    return generate_gif

def find_corruption(output_hist, input_hist, closest_idx_hist, path_points, path_headings, velocity_profile, Cs, N, starting_k, optimizer, model_params, noise_std, model_at_idx, desired_output_fn, normalize_output):
    """
    Finds the corruption in the given data.

    Parameters:
        output_hist: list[np.ndarray] - a list of N outputs
        input_hist: list[np.ndarray] - a list of N-1 inputs
        closest_idx_hist: list[int] - a list of N-1 closest indices on the path
        path_points: list[tuple[float,float]] - a list of points that describe the path
        path_headings: list[float] - a list of headings at each point on the path
        velocity_profile: list[float] - a list of velocities at each point on the path
        Cs: list[np.ndarray] - a list of N output matrices
        N: int - the time window
        starting_k: int - the starting time step to search from
        optimizer: MyOptimizer - the optimizer to use
        model_params: dict - the model parameters
        noise_std: float - the standard deviation of the measurement noise
    """
    # Run the solver from the beginning until we detect the corruption
    ks = range(max(N,starting_k), len(output_hist)+1)
    args_iterable = (
        (
            None,
            output_hist[k-N:k],
            input_hist[k-N:k-1],
            closest_idx_hist[k-N:k-1],
            path_points,
            path_headings,
            velocity_profile,
            Cs[k-N:k],
            model_params['dt'],
            model_params['l'],
            N,
            True,
            optimizer,
            noise_std,
            model_at_idx,
            desired_output_fn,
            normalize_output,
            False, # disable estimator
        )
        for k in ks
    )

    pool = None
    res_iter = map(estimate_state_unpack, args_iterable)
    # Optional parallelization
    # pool = Pool(min(MAX_POOL_SIZE, os.cpu_count() or 1))
    # pool_chunksize = int(1/model_params['dt']) # set chunksize to be a deterministic number of seconds
    # res_iter = pool.imap(estimate_state_unpack, args_iterable, chunksize=pool_chunksize)

    try:
        # Run the solver from the beginning until we detect the corruption
        for k, (estm_res, optimizer_res, metadata) in zip(ks, res_iter):
            # print(f"{k=} (t={k*model_params['dt']:.2f}s)")
            if optimizer_res is None:
                continue

            soln, solns, optimizer_metadata = optimizer_res
            if soln is None:
                continue

            x0_hat, prob, soln_metadata = soln
            if len(soln_metadata['K']) > 0:
                return {
                    'k': k,
                    't': k*model_params['dt'],
                    'K': soln_metadata['K'],
                    'metadata': metadata,
                    'optimizer_metadata': optimizer_metadata,
                }
        else:
            return None
    finally:
        if pool:
            pool.terminate()
            pool.join()

def run_experiment(
    # Simulation parameters
    x0,
    C,
    noise_std,
    num_steps,
    N,
    path_points,
    path_headings,
    path_curvatures,
    path_dcurvatures,
    velocity_profile,
    optimizer,
    model_params,
    real_time_fault_tolerance,
    # Fault specification
    fault_spec,
):
    fault = getattr(fault_generators, fault_spec['fn'])(**fault_spec['kwargs'])

    simulation = {}
    start = time.perf_counter()
    try:
        (
            t_hist,
            state_hist,
            output_hist,
            estimate_hist,
            u_hist,
            closest_idx_hist,
            ukf_P_hist,
        ) = run_simulation(
            x0,
            C,
            noise_std,
            num_steps,
            N,
            path_points,
            path_headings,
            path_curvatures,
            path_dcurvatures,
            velocity_profile,
            optimizer,
            model_params,
            fault,
            real_time_fault_tolerance,
        )
    except Exception as e:
        simulation['error'] = str(e)
        simulation['stacktrace'] = traceback.format_exc()
        return simulation, None
    finally:
        simulation['duration'] = time.perf_counter() - start

    # Post-analysis
    corruption = find_corruption(
        output_hist,
        u_hist,
        closest_idx_hist,
        path_points,
        path_headings,
        velocity_profile,
        [C] * len(output_hist),
        N,
        500,
        optimizer,
        model_params,
        noise_std,
        model_at_idx=lambda idx: kinematic_bicycle_model_linearize(path_headings[idx], velocity_profile[idx], 0, model_params['dt'], model_params['l']),
        desired_output_fn=lambda i, idx: C @ kinematic_bicycle_model_desired_state_at_idx(idx, path_points, path_headings, velocity_profile),
        normalize_output=kinematic_bicycle_model_normalize_output,
    )

    return simulation, corruption


def run_experiment_unpack(args):
    return run_experiment(*args)

def run_experiments(
    output_path,
    # Simulation parameters
    x0,
    C,
    noise_std,
    num_steps,
    N,
    path_points,
    path_headings,
    path_curvatures,
    path_dcurvatures,
    velocity_profile,
    optimizer,
    model_params,
    real_time_fault_tolerance,
    # Fault specification
    fault_specs,
    extra_output_metadata={},
):
    # Create the output file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.touch()

    # Generate faults
    exp_args = (
        (
            # Simulation parameters
            x0,
            C,
            noise_std,
            num_steps,
            N,
            path_points,
            path_headings,
            path_curvatures,
            path_dcurvatures,
            velocity_profile,
            optimizer,
            model_params,
            real_time_fault_tolerance,
            # Fault specification
            fault_spec,
        ) for fault_spec in fault_specs
    )
    
    # Parallelized
    pool = Pool(min(MAX_POOL_SIZE, os.cpu_count() or 1))
    res_iter = zip(pool.imap(run_experiment_unpack, exp_args), fault_specs)
    # Single-threaded
    # pool = None
    # res_iter = zip(map(run_experiment_unpack, exp_args), fault_specs)
    try:
        for (simulation, corruption), fault_spec in tqdm(res_iter, total=len(fault_specs), smoothing=0):
            # write results to file
            with open(output_file, 'a') as f:
                f.write(json.dumps({
                    'fault_spec': fault_spec,
                    'simulation': simulation,
                    'corruption': corruption,
                    **extra_output_metadata,
                }, cls=NpEncoder)+"\n")
    finally:
        if pool:
            pool.terminate()
            pool.join()

def arange_inclusive(start, stop, step):
    """
    Like np.arange, but includes the stop value.
    """
    return np.arange(start, stop + step, step)