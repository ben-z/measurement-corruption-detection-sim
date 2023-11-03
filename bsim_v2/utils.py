import cvxpy as cp
import numpy as np
import os
import time
from collections import namedtuple
from itertools import chain, combinations, repeat
from math import pi, sin, cos, atan2, sqrt
from numpy.typing import NDArray
from numpy.linalg import matrix_power
from typing import TypeVar, Iterable, Tuple, Optional
from scipy.linalg import expm
from multiprocessing import Pool

MAX_POOL_SIZE = 32

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
    assert len(path_points) == len(velocities), "path_points and velocities must have the same length"
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

def optimize_l0_v2(Phi: np.ndarray, Y: np.ndarray, eps: NDArray[np.float64] | float = 1e-15, S_list: Optional[Iterable[Iterable[int]]] = None, solver_args: dict = {}):
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

    N, q, n = Phi.shape
    assert Y.shape == (N, q)

    cvx_Y = Y.reshape((N*q,)) # groups of sensor measurements stacked vertically
    cvx_Phi = Phi.reshape((N*q, n)) # groups of transition+output matrices stacked vertically

    # Support scalar or vector eps
    if np.isscalar(eps):
        eps_final: NDArray[np.float64] = np.ones(q) * eps
    else:
        eps_final: NDArray[np.float64] = eps

    def optimize_case(S):
        """
        Solves the l0 minimization problem for a given set of uncorrupted sensors $S$.
        """
        # K is the sensors that can be corrupted (i.e. the sensors that are not in S)
        K = list(set(range(q)) - set(S))

        x0_hat = cp.Variable(n)
        optimizer = cp.reshape(cvx_Y - np.matmul(cvx_Phi, x0_hat), (q, N))
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

    solns = [optimize_case(S) for S in S_list]
    for x0_hat, prob, metadata in solns:
        if prob.status in ["optimal", "optimal_inaccurate"]:
            return (x0_hat, prob, metadata, solns)

    return (None, None, None, solns)

# Stripped down cvxpy.Problem that is serializable
MyCvxpyProblem = namedtuple('MyCvxpyProblem', ['status', 'solver_stats', 'compilation_time', 'solve_time', 'value'])

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
    
    def optimize_l0_v4(self, Phi: np.ndarray, Y: np.ndarray, eps: NDArray[np.float64] | float = 1e-15, S_list: Optional[Iterable[Iterable[int]]] = None, solver_args: dict = {}):
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
        start = time.perf_counter()

        N, q, n = Phi.shape
        assert Y.shape == (N, q)
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
        print(f"Setup time: {end-start:.4f}s")

        map_args = [S_list, repeat(q), repeat(self.prob), repeat(self.x0_hat), repeat(self.can_corrupt), repeat({'solver': self.solver, **solver_args})]
        soln_generator = map(optimize_l0_case, *map_args)
        # with Pool(min(MAX_POOL_SIZE, os.cpu_count() or 1)) as pool:
        #   soln_generator = pool.starmap(optimize_l0_case, zip(*map_args))

        solns = []
        for soln in soln_generator:
            x0_hat, prob, metadata = soln
            solns.append(soln)
            if prob.status in ["optimal", "optimal_inaccurate"]:
                return (x0_hat, prob, metadata, solns)

        return (None, None, None, solns)

def optimize_l0_case(S: Iterable[int], q: int, prob: cp.Problem, x0_hat: cp.Variable, can_corrupt: cp.Parameter, solver_args: dict = {}):
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
    # K is the sensors that can be corrupted (i.e. the sensors that are not in S)
    K = list(set(range(q)) - set(S))

    can_corrupt.value = np.ones(q)
    for j in S:
        can_corrupt.value[j] = False

    start = time.perf_counter()
    prob.solve(**solver_args)
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