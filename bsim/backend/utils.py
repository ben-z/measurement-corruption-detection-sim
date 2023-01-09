from numpy.linalg import matrix_power
import cvxpy as cp
from itertools import chain, combinations
import json
import numpy as np
from multiprocessing.pool import ThreadPool as Pool
import time
from typing import Dict, Any
from functools import cached_property, lru_cache, wraps, update_wrapper

def closest_point_on_line_segment(p, a, b):
    """
    Find the closest point on the line segment defined by a and b to the point p.
    Also returns the parameter t, which is the progress [0,1] along the line segment ab.
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
    Also returns the parameter t, which is the progress ([0,1] if on the line segment)
    along the line ab.
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


def calc_input_effects_on_output(A, B, C, inputs):
    # calculates the input effects on the output
    # A: numpy.ndarray - matrix of size (n,n)
    # B: numpy.ndarray - matrix of size (n,m)
    # C: numpy.ndarray - matrix of size (p,n)
    # inputs: numpy.ndarray - matrix of size (m,N)
    # returns: numpy.ndarray - matrix of size (p,N)

    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]
    N = inputs.shape[1]

    assert A.shape == (n, n)
    assert B.shape == (n, m)
    assert C.shape == (p, n)
    assert inputs.shape == (m, N)

    # TODO: make this faster by reusing the calculations for Phi
    effects = np.zeros((p, N))
    for k in range(N):
        for j in range(k):
            effects[:, k] += C @ matrix_power(A, k-1-j) @ B @ inputs[:, j]

    return effects


def optimize_l1(n, p, N, Phi, Y):
    # solves the l1/l2 norm minimization problem
    # n: int - number of states
    # p: int - number of outputs
    # N: int - number of time steps
    # Phi: numpy.ndarray - matrix of size (p*N, n) - (C*A^0, C*A^1, ..., C*A^(N-1))'
    # Y: numpy.ndarray - measured outputs, with input effects subtracted, size (p*N)
    # returns: numpy.ndarray

    assert Phi.shape == (p*N, n)
    assert Y.shape == (p*N,)

    x0_hat = cp.Variable(n)
    # define the expression that we want to run l1/l2 optimization on
    optimizer = Y - np.matmul(Phi, x0_hat)
    # reshape to adapt to l1/l2 norm formulation
    # Note that cp uses fortran ordering (column-major), this is different from numpy,
    # which uses c ordering (row-major)
    optimizer_reshaped = cp.reshape(optimizer, (p, N))
    optimizer_final = cp.mixed_norm(optimizer_reshaped, p=2, q=1)
    # Equivalent to optimizer_final = cp.norm1(cp.norm(optimizer_reshaped, axis=1))

    obj = cp.Minimize(optimizer_final)

    # Form and solve problem.
    prob = cp.Problem(obj)
    prob.solve(verbose=True)  # Returns the optimal value.

    return (prob, x0_hat)


def optimize_l0(n, p, N, Phi, Y, eps: np.ndarray = 0.2):
    # solves the l0 minimization problem
    # n: int - number of states
    # p: int - number of outputs
    # N: int - number of time steps
    # Phi: numpy.ndarray - matrix of size (p*N, n) - (C*A^0, C*A^1, ..., C*A^(N-1))'
    # Y: numpy.ndarray - measured outputs, with input effects subtracted, size (p*N)
    # returns: numpy.ndarray

    assert Phi.shape == (p*N, n)
    assert Y.shape == (p*N,)

    # with Pool(processes=30) as pool:
    #     solns = pool.starmap(optimize_l0_subproblem, [(n, p, N, Phi, Y, attacked_sensor_indices) for attacked_sensor_indices in powerset(range(p))])
    solns = [optimize_l0_subproblem(n, p, N, Phi, Y, attacked_sensor_indices, eps) for attacked_sensor_indices in powerset(range(p)) if len(attacked_sensor_indices) <= 1]
    for prob, x_hat_l0 in solns:
    # for attacked_sensor_indices in powerset(range(p)):
        # prob, x_hat_l0 = optimize_l0_subproblem(n, p, N, Phi, Y, attacked_sensor_indices, eps)
        if prob.status in ["optimal", "optimal_inaccurate"]:
            return (prob, x_hat_l0)

    raise Exception("No solution found")


def optimize_l0_subproblem(n, p, N, Phi, Y, attacked_sensor_indices, eps):
    x0_hat = cp.Variable(n)
    optimizer = Y - np.matmul(Phi, x0_hat)
    optimizer_reshaped = cp.reshape(optimizer, (p, N))
    optimizer_final = cp.mixed_norm(optimizer_reshaped, p=2, q=1)

    # Support both scalar eps and eps per sensor
    if np.isscalar(eps):
        eps = np.ones(p) * eps

    constraints = []
    for j in set(range(p)) - set(attacked_sensor_indices):
        for t in range(N):
            # constraints.append(cp.norm(optimizer[p*t+j]) <= eps)
            constraints.append(optimizer[p*t+j] <= eps[j])
            constraints.append(optimizer[p*t+j] >= -eps[j])
    
    prob = cp.Problem(cp.Minimize(cp.norm(optimizer_final)), constraints)
    start = time.time()
    prob.solve()
    end = time.time()

    print(f"Solved l0 subproblem in {end-start:.2f} seconds. Indices: {attacked_sensor_indices}, status: {prob.status}, value: {prob.value}")

    return (prob, x0_hat)


def get_l0_state_estimation_l2_bound(A: np.ndarray, C: np.ndarray, largest_sensor_deviations: np.ndarray, qmax: int, N: int):
    """
    Calculates the l0 state estimation bound for a given system and sensor deviations.
    """
    n = A.shape[0]
    p = C.shape[0]

    Rs = [list(s) for s in powerset(range(p)) if len(s) == p - 2*qmax]

    bound = -np.inf

    for R in Rs:
        max_noise_norm = np.linalg.norm(largest_sensor_deviations[R])

        O_R = get_observability_matrix(A, C, R, N)

        pinv_O_R = np.linalg.pinv(O_R)

        largest_singular_value = max(np.linalg.svd(pinv_O_R, compute_uv=False))

        bound = max(bound, largest_singular_value * 2*max_noise_norm)

    return bound

def matrix_spectral_norm(A):
    return max(np.linalg.eigvals(A))

def get_error_estimation_l2_bounds(A: np.ndarray, C: np.ndarray, Dx: float, largest_sensor_deviations: np.ndarray, N: int):
    """
    Calculates the error estimation bounds for a given system and sensor deviations.
    """
    n = A.shape[0]
    p = C.shape[0]

    bounds = np.zeros(p)

    for i in range(p):
        O_i = np.zeros((N, n))
        for t in range(N):
            O_i[t, :] = C[i, :] @ matrix_power(A, t)
        
        largest_singular_value = max(np.linalg.svd(O_i, compute_uv=False))

        bounds[i] = largest_singular_value * Dx + 2*largest_sensor_deviations[i]
    
    return bounds

def s_sparse_observability(A,C):
    """
    Calculates the s-sparse observability of a system.
    """
    n = A.shape[0]
    p = C.shape[0]

    s = -1
    for K in powerset(range(p)):
        # K is the attacked sensors
        remaining_sensors = list(set(range(p)) - set(K))

        O_K = get_observability_matrix(A, C, remaining_sensors, n) # C with only sensors in K^c
        
        O_K_rank = np.linalg.matrix_rank(O_K)

        if s == -1 and O_K_rank < n:
            # found the first non-full-rank entry
            s = len(K) - 1
            break

    return s


def get_observability_matrix(A, C, R=None, N=None):
    """
    Calculates the observability matrix for a given system (A, C) and sensor set R after N time steps.
    """
    n = A.shape[0]
    p = C.shape[0]

    if R is None:
        R = range(p)

    if N is None:
        N = n

    # C with only sensors in R
    PC = C[list(R), :]

    # Calculate O_R
    O_R = np.zeros((len(R)*N, n))
    for t in range(N):
        O_R[t*len(R):(t+1)*len(R), :] = PC @ matrix_power(A, t)

    return O_R

def is_l1_state_estimation_error_bounded(A, C, K, N=None):
    """
    Checks if the l1 state estimation error is bounded for a given system (A, C) and set of attacked sensors K.
    """
    n = A.shape[0]
    p = C.shape[0]

    if N is None:
        N = n

    K_c = list(set(range(p)) - set(K))

    O_K = get_observability_matrix(A, C, K, N)
    O_K_c = get_observability_matrix(A, C, K_c, N)

    # TODO: find a way to check equation 25 for all x in the workspace.
    pass

def does_l1_state_estimation_error_analytical_bound_hypothesis_hold_for_K(A, C, K, N=None):
    """
    Checks if the l1 state estimation error analytical bound hypothesis (Pajic, 2017, eqn 29) holds
    for the system (A,C) and set of attacked sensors K. Note that equation 29 has to be true for all
    K of size q for the exact theorem.
    """
    n = A.shape[0]
    p = C.shape[0]
    q = len(K)

    if N is None:
        N = n

    K_c = list(set(range(p)) - set(K))

    O_K = get_observability_matrix(A, C, K, N)
    O_K_c = get_observability_matrix(A, C, K_c, N)

    return (O_K_c.T @ O_K_c - q * N**2 * O_K.T @ O_K >= np.finfo(float).eps * np.eye(n)).all()

def get_l1_state_estimation_l2_bound(A: np.ndarray, C: np.ndarray, largest_sensor_deviations: np.ndarray, K: np.ndarray, N: int):
    """
    Calculates the l1 state estimation bound for a given system and sensor deviations.
    FIXME: CVX errors out on this formulation. The message is "Problem does not follow DCP rules."
    """
    n = A.shape[0]
    p = C.shape[0]

    K_c = list(set(range(p)) - set(K))

    Os = [get_observability_matrix(A, C, [s], N) for s in range(p)]

    sigma = np.linalg.norm(largest_sensor_deviations)

    dx1_l1 = cp.Variable(n)
    optimizer = cp.norm(dx1_l1)
    constraint_lhs = sum(cp.norm(Os[i] @ dx1_l1) for i in K_c)
    constraint_rhs = sum(cp.norm(Os[i] @ dx1_l1) for i in K) + 2*sigma
    constraints = [constraint_lhs <= constraint_rhs]

    prob = cp.Problem(cp.Maximize(optimizer), constraints)
    start = time.time()
    prob.solve()
    end = time.time()

    print(f"Solved l1 state estimation bound in {end-start:.2f} seconds. status: {prob.status}, value: {prob.value}")

    return (prob, dx1_l1)

class SegmentInfoItem:
    # _p0 and _p1 are the endpoints of the segment
    # they are immutable once set. This is achieved
    # by using the read-only properties p0 and p1.
    _p0: np.ndarray
    _p1: np.ndarray
    progress: float

    def __init__(self, p0, p1, progress=np.nan) -> None:
        self._p0 = p0
        self._p1 = p1
        self.progress = progress
    
    def __deepcopy__(self, memo):
        return SegmentInfoItem(self.p0, self.p1, self.progress)
    
    def to_dict(self):
        return {
            "p0": self.p0,
            "p1": self.p1,
            "progress": self.progress
        }
    
    def set_progress(self, progress):
        self.progress = progress

    @property
    def p0(self):
        return self._p0
    
    @property
    def p1(self):
        return self._p1

    @cached_property
    def length(self):
        return np.linalg.norm(self.p1 - self.p0)

    @cached_property
    def heading(self):
        return np.arctan2(self.p1[1] - self.p0[1], self.p1[0] - self.p0[0])

    @property
    def distance_travelled(self):
        return self.progress * self.length
    
    @property
    def distance_remaining(self):
        return (1 - self.progress) * self.length
    
    @property
    def closest_point(self):
        return self.p0 + self.progress * (self.p1 - self.p0)
    
def generate_segment_info(pos: np.ndarray, path_points: np.ndarray, wrap=True):
    """
    Generates the segment information (closest point, distance travelled, etc.)
    for a given path. Assumes linear interpolation between path points.
    pos: the current position
    path_points: the points on the path, the last point is assumed to be
        connected to the first point to form a closed path.
    wrap: whether to wrap around the path or not.
    """

    # pairs of points that form the path: [(p0, p1), (p1, p2), ..., (pn, p0)]
    path_segments = np.stack([path_points, np.roll(path_points, -1, axis=0)], axis=1)
    if not wrap:
        path_segments = path_segments[:-1]

    # segment_info is used to make decisions about which segment to use
    segment_info = []
    for p0, p1 in path_segments:
        _, progress = closest_point_on_line_segment(pos, p0, p1)

        segment_info.append(SegmentInfoItem(p0, p1, progress))

    return segment_info

class EndOfPathError(ValueError):
    pass

def move_along_path(segment_info, current_path_segment_idx, step_size_m, wrap=True):
    """
    Moves along the path by step_size_m starting from the point determined by
    current_path_segment_idx and the corresponding progress. Mutates segment_info
    and returns the updated info for the target segment.
    """
    remaining_m = step_size_m

    while True:
        current_path_segment = segment_info[current_path_segment_idx]
        if (remaining_m >= 0 and current_path_segment.distance_remaining >= remaining_m) or \
            (remaining_m < 0 and current_path_segment.distance_travelled >= -remaining_m):
            # we don't travel enough to switch to the next/prev segment
            current_path_segment.set_progress(current_path_segment.progress + remaining_m / current_path_segment.length)
            yield current_path_segment, current_path_segment_idx
            remaining_m = step_size_m
        elif remaining_m >= 0:
            # we travel to the next segment
            remaining_m -= current_path_segment.distance_remaining
            current_path_segment.set_progress(1)
            
            # current_path_segment_idx = (current_path_segment_idx + 1) % len(segment_info)
            current_path_segment_idx += 1
            if current_path_segment_idx >= len(segment_info):
                if wrap:
                    current_path_segment_idx = 0
                else:
                    raise EndOfPathError("Cannot move forward along path, reached end of path")
            segment_info[current_path_segment_idx].set_progress(0)
        elif remaining_m < 0:
            # we travel to the previous segment
            remaining_m += current_path_segment.distance_travelled
            current_path_segment.set_progress(0)
            
            current_path_segment_idx -= 1
            if current_path_segment_idx < 0:
                if wrap:
                    current_path_segment_idx = len(segment_info) - 1
                else:
                    raise EndOfPathError("Cannot move backward along path, reached end of path")
            segment_info[current_path_segment_idx].set_progress(1)
        else:
            raise Exception(f"This shouldn't happen! remaining_m: {remaining_m}")


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

class JSONNumpyDecoder(json.JSONDecoder):
    """
    Decodes numerical lists as numpy lists.
    Example usage:
    ```python
    data = json.loads(json_string, cls=JSONNumpyDecoder)
    ```
    """

    def decode(self, s):
        data = super().decode(s)
        return self._decode(data)

    def _decode(self, data):
        if isinstance(data, int):
            # There are no `int`s in JavaScript: https://stackoverflow.com/a/16662153/4527337
            return float(data)
        if isinstance(data, (float, str)):
            return data
        elif isinstance(data, dict):
            return {self._decode(key): self._decode(value) for key, value in data.items()}
        elif isinstance(data, list):
            decoded = [self._decode(element) for element in data]
            npdecoded = np.array(decoded)
            # if the list is a numpy array, return it as a numpy array
            if npdecoded.dtype != object:
                return npdecoded
            else:
                return decoded
        elif data == 'NaN':
            return np.nan
        elif data == 'inf':
            return np.inf
        elif data == '-inf':
            return -np.inf
        else:
            raise ValueError('Unknown data type: {}'.format(data))

def frenet2global_point(x, y, theta, s, d):
    """
    Converts a point in frenet coordinates to global coordinates.
    x, y, theta: the global coordinates of the reference point
    s, d: the frenet coordinates of the point
    """
    x += s * np.cos(theta) - d * np.sin(theta)
    y += s * np.sin(theta) + d * np.cos(theta)
    return x, y

def ensure_options_are_known(options: Dict[str, Any], known_options: Dict[str, Any], name: str = ""):
    """
    Ensures that all options in options are in known_options.
    """
    unknown_options = set(options) - set(known_options)
    if unknown_options:
        raise ValueError(f"Unknown {f'{name} ' if name else ''}options: {unknown_options}")

class PerfCounter:
    """
    A simple performance counter that can be used to measure the time spent in
    a block of code. Example usage:
    ```python
    with PerfCounter() as pc:
        # do something
    print(pc.elapsed_)
    ```
    """
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()

    @property
    def elapsed_s(self):
        assert self.start_time is not None
        assert self.end_time is not None
        return (self.end_time - self.start_time)

class AutoPerfCounter(PerfCounter):
    """
    A performance counter that automatically stores the elapsed time in a
    provided dictionary.
    ```python
    execution_times = {}
    with AutoPerfCounter(execution_times, "my_block"):
        # do something
    print(f"my_block took {execution_times['my_block']}s")
    ```
    """
    def __init__(self, execution_times: Dict[str, float], name: str):
        super().__init__()
        self.execution_times = execution_times
        self.name = name

    def __exit__(self, *args):
        super().__exit__(*args)
        self.execution_times[self.name] = self.elapsed_s

def get_evolution_matrices(As, Cs):
    """
    Inputs:
    As: Discrete-time A matrices
    Cs: Discrete-time C matrices

    Returns:
    state_evolution_matrix = [
        I,
        As[0],
        As[1]@As[0],
        ...
    ]
    output_evolution_matrix = [
        Cs[0],
        Cs[1]@As[0],
        Cs[2]@As[1]@As[0],
        ...
    ]
    In this way,
    state_evolution_matrix@x0 will return a vector that contains all states over time.
    output_evolution_matrix@x0 will return a vector that contains all measurements
    defined by the measurement matrices in Cs.

    Restrictions:
    1. len(As) == len(Cs) - 1
    2. elements of As and Cs are of the correct dimensions
       (e.g A has (n x n) elements, C[k] has (q_k x n) elements).
    """

    assert len(As) == len(Cs) - 1
    assert len(As) > 0

    n = As[0].shape[0]
    state_evolution_matrix_list = [np.eye(n)]
    for A_k in As:
        prev_matrix = state_evolution_matrix_list[-1]
        state_evolution_matrix_list.append(A_k@prev_matrix)

    output_evolution_matrix_list = [C_k @ m for m, C_k in zip(state_evolution_matrix_list, Cs)]

    return np.concatenate(state_evolution_matrix_list), np.concatenate(output_evolution_matrix_list)

def make_hashable(a):
    if isinstance(a, np.ndarray):
        return a.tobytes()
    elif isinstance(a, list):
        return tuple(make_hashable(b) for b in a)
    return a

class HashableIterable:
    """
    A wrapper for an iterable that makes it hashable. Specifically, it converts
    all numpy arrays to bytes and stores the result as a tuple. This is useful
    for caching functions that take numpy arrays as arguments.
    """

    def __init__(self, iterable):
        self.raw = iterable
        self.hashable = tuple(make_hashable(a) for a in iterable)
    
    
    def __hash__(self):
        return hash(self.hashable)
    
    def __eq__(self, other):
        return self.hashable == other.hashable
    
    def __repr__(self):
        return repr(self.raw)
    
    def get_raw(self):
        return self.raw

def np_cache(*cache_args,**cache_kwargs):
    """
    Decorator that caches the result of a function that takes numpy arrays as
    arguments.
    """
    def decorating_function(user_function):
        @lru_cache(*cache_args,**cache_kwargs)
        def cached_wrapper(hashable):
            return user_function(*hashable.get_raw())

        @wraps(user_function)
        def wrapper(*args):
            hashable = HashableIterable(args)

            return cached_wrapper(hashable)

        return wrapper

    return decorating_function


@np_cache(maxsize=200)
def get_observability_mapping(C, sensor_configurations=None):
    """
    This function solves the observability problem for a static system
    y = Cx.
    Inputs:
        C: numpy.ndarray - matrix of size (p,n)
        sensor_configurations: int[][] - list of sensor configurations. Each
            configuration is a list of sensor indices. If None, all possible
            configurations are considered.
    Outputs:
        mapping: {int => int[][]} - mapping from state indices to sensor configurations
            that uniquely recover the state.
    """
    p = C.shape[0]
    n = C.shape[1]
    if sensor_configurations is None:
        using_all_sensor_configurations = True
        sensor_configurations = powerset(range(p))
    else:
        using_all_sensor_configurations = False

    mapping = {}

    configs = 0
    small_enough_configs = 0
    rank_ops = 0
    redundant_configs = 0
    new_configs = 0
    redundant_useful_rank_ops = 0
    useful_rank_ops = 0

    for S in sensor_configurations:
        configs += 1

        if len(S) == 0:
            # no sensors selected
            continue

        if using_all_sensor_configurations and len(S) > n:  
            # if we are using all sensor configurations, then we don't need to consider
            # configurations that have more sensors than states, because the rank of
            # C_S will never be equal to len(S)
            continue

        small_enough_configs += 1

        # C matrix with the select sensors
        C_S = C[S, :]
        
        # Set of indices indicating non-zero columns of C_s
        # this is the states that are affected by the sensors in S
        affected_states = np.asarray(np.any(C_S, axis=0)).nonzero()[-1]

        if using_all_sensor_configurations and len(affected_states) != len(S):
            # if len(S) > len(affected_states), then there are redundant roles in C_S, so S is not minimal
            # if len(S) < len(affected_states), then the rank of C_S will never be equal to len(affected_states)
            continue

        rank_ops += 1
        C_S_rank = np.linalg.matrix_rank(C_S)

        # if C_S has a left inverse for the affected states, then the states in affected_states
        # can be uniquely recovered by the sensors in S.
        if C_S_rank == len(affected_states):
            S_set = frozenset(S)

            useful_rank_ops += 1

            curr_new_configs = new_configs

            for affected_state in affected_states:
                # record S if it doesn't already contain a set of sensors that
                # can recover the affected state
                if not any(s.issubset(S_set) for s in mapping.get(affected_state, [])):
                    new_configs += 1
                    mapping.setdefault(affected_state, []).append(S_set)
                else:
                    redundant_configs += 1
            
            if curr_new_configs == new_configs:
                # redundant matrix rank operation
                redundant_useful_rank_ops += 1

    # Debug print
    # print(f"{configs=} {small_enough_configs=} {rank_ops=} {redundant_configs=} {new_configs=} {useful_rank_ops=} {redundant_useful_rank_ops=}")
    
    return mapping


def is_observable(mapping=None, C=None, missing_sensors=[], sensor_configurations=None):
    """
    This function checks if a static system is observable given a set of missing sensors.
    Inputs:
        mapping: {int => int[][]} - mapping from state indices to sensor configurations
            that uniquely recover the state. If None, this is computed from C.
        C: numpy.ndarray - matrix of size (p,n). This is only used if mapping is None.
        missing_sensors: int[] - list of sensor indices that are missing.
        sensor_configurations: int[][] - list of possible sensor configurations. If None,
            all configurations will be considered.
    Outputs:
        is_observable: bool - True if the system is observable with the missing sensors, False otherwise.
    """

    if mapping is None:
        mapping = get_observability_mapping(C, sensor_configurations)
    
    for _state, sensor_configurations in mapping.items():
        uncompromised_configurations = [S for S in sensor_configurations if not set(S).intersection(missing_sensors)]
        # if there is no configuration that can uniquely recover the state, then the system is not observable
        if len(uncompromised_configurations) == 0:
            return False
    
    return True

def is_attackable(mapping=None, C=None, attacked_sensors=[], sensor_configurations=None):
    """
    This function checks if a static system is attackable given a set of attacked sensors.
    Inputs:
        mapping: {int => int[][]} - mapping from state indices to sensor configurations
            that uniquely recover the state. If None, this is computed from C.
        C: numpy.ndarray - matrix of size (p,n). This is only used if mapping is None.
        attacked_sensors: int[] - list of sensor indices that are attacked.
        sensor_configurations: int[][] - list of possible sensor configurations. If None,
            all configurations will be considered.
    Outputs:
        is_attackable: bool - True if the system is attackable with the attacked sensors, False otherwise.
    """

    if mapping is None:
        mapping = get_observability_mapping(C, sensor_configurations)

    for _state, sensor_configurations in mapping.items():
        compromised_configurations = [S for S in sensor_configurations if set(S).intersection(attacked_sensors)]
        # if at least half of the configurations are compromised, then we cannot recover the state
        if len(compromised_configurations) >= len(sensor_configurations) / 2:
            return False
    
    return True

def expand_sensor_configs_over_time(sensor_configurations, p, N):
    """
    This function turns a set of sensor configurations into a set of sensor configurations
    useful when turning a system into a static problem.
    sensor_configurations: int[][] - list of sensor configurations
    p - number of sensors
    N - number of time steps

    Outputs:
        expanded_sensor_configurations: int[][] - list of sensor configurations
    
    Example:
        sensor_configurations = [[0,1], [2,3]]
        p = 4
        N = 2
        expanded_sensor_configurations = [[0,1,4,5], [2,3,6,7]]
    """

    return [scs + tuple(t*p+i for t in range(N) for i in scs) for scs in sensor_configurations]
    

def is_observable_ltv(Cs=None, As=None, missing_sensors=[], sensor_configurations=None):
    """
    This function checks if a discrete linear time-varying system is observable given a set of missing sensors.
    Continuous time systems need to be discretized before calling this function.

    Inputs:
        mapping: {int => int[][]} - mapping from state indices to sensor configurations
            that uniquely recover the state. If None, this is computed from C.
        Cs: numpy.ndarray[] - array of matrix of size (p,n). This is only used if mapping is None.
        As: numpy.ndarray[] - array of matrix of size (n,n). This is only used if mapping is None.
        missing_sensors: int[] - list of sensor indices that are missing.
        sensor_configurations: int[][] - list of possible sensor configurations. If None,
            all configurations will be considered.
    Outputs:
        is_observable: bool - True if the system is observable with the missing sensors, False otherwise.
    """

    assert Cs is not None
    assert As is not None
    assert len(As) == len(Cs) - 1

    p = Cs[0].shape[0]
    N = len(Cs) # the number of time steps

    # output_evolution_matrix is a matrix C where Y = C@x_0
    output_evolution_matrix = get_evolution_matrices(Cs=Cs, As=As)[1]
    output_evolution_matrix.setflags(write=False)

    if sensor_configurations is None:
        sensor_configurations = powerset(range(p))

    return is_observable(C=output_evolution_matrix, missing_sensors=[t*p+i for t in range(N) for i in missing_sensors], sensor_configurations=expand_sensor_configs_over_time(sensor_configurations, p, N))

def is_attackable_ltv(Cs=None, As=None, attacked_sensors=[], sensor_configurations=None):
    """
    This function checks if a discrete linear time-varying system is attackable given a set of attacked sensors.
    Continuous time systems need to be discretized before calling this function.

    Inputs:
        mapping: {int => int[][]} - mapping from state indices to sensor configurations
            that uniquely recover the state. If None, this is computed from C.
        Cs: numpy.ndarray[] - array of matrix of size (p,n). This is only used if mapping is None.
        As: numpy.ndarray[] - array of matrix of size (n,n). This is only used if mapping is None.
        attacked_sensors: int[] - list of sensor indices that are attacked.
        sensor_configurations: int[][] - list of possible sensor configurations. If None,
            all configurations will be considered.
    Outputs:
        is_attackable: bool - True if the system is attackable with the attacked sensors, False otherwise.
    """

    assert Cs is not None
    assert As is not None
    assert len(As) == len(Cs) - 1

    p = Cs[0].shape[0]
    N = len(Cs) # the number of time steps

    # output_evolution_matrix is a matrix C where Y = C@x_0
    output_evolution_matrix = get_evolution_matrices(Cs=Cs, As=As)[1]

    return is_attackable(C=output_evolution_matrix, attacked_sensors=[t*p+i for t in range(N) for i in attacked_sensors], sensor_configurations=expand_sensor_configs_over_time(sensor_configurations, p, N))

# TODO: Check if is_attackable is a generalization of 2s-sparse observability