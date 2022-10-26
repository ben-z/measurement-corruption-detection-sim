from numpy.linalg import matrix_power
import cvxpy as cp
from itertools import chain, combinations
import json
import numpy as np
from multiprocessing.pool import ThreadPool as Pool
import time

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
    for attacked_sensor_indices in powerset(range(p)):
        prob, x_hat_l0 = optimize_l0_subproblem(n, p, N, Phi, Y, attacked_sensor_indices, eps)
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
    def __init__(self, p0, p1, progress=np.nan) -> None:
        self.p0 = p0
        self.p1 = p1
        self.length = np.linalg.norm(p1 - p0)
        self.heading = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
        self.progress = progress
    
    def set_progress(self, progress):
        self.progress = progress

    @property
    def distance_travelled(self):
        return self.progress * self.length
    
    @property
    def distance_remaining(self):
        return (1 - self.progress) * self.length
    
    @property
    def closest_point(self):
        return self.p0 + self.progress * (self.p1 - self.p0)
    
def generate_segment_info(pos: np.ndarray, path_points: np.ndarray):
    """
    Generates the segment information (closest point, distance travelled, etc.)
    for a given path. Assumes linear interpolation between path points.
    pos: the current position
    path_points: the points on the path, the last point is assumed to be
        connected to the first point to form a closed path.
    """

    # pairs of points that form the path: [(p0, p1), (p1, p2), ..., (pn, p0)]
    path_segments = np.stack([path_points, np.roll(path_points, -1, axis=0)], axis=1)

    # segment_info is used to make decisions about which segment to use
    segment_info = []
    for p0, p1 in path_segments:
        _, progress = closest_point_on_line_segment(pos, p0, p1)

        segment_info.append(SegmentInfoItem(p0, p1, progress))

    return segment_info

def move_along_path(segment_info, current_path_segment_idx, step_size_m):
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
            yield current_path_segment
            remaining_m = step_size_m
        elif remaining_m >= 0:
            # we travel to the next segment
            remaining_m -= current_path_segment.distance_remaining
            current_path_segment.set_progress(1)
            
            current_path_segment_idx = (current_path_segment_idx + 1) % len(segment_info)
            segment_info[current_path_segment_idx].set_progress(0)
        elif remaining_m < 0:
            # we travel to the previous segment
            remaining_m += current_path_segment.distance_travelled
            current_path_segment.set_progress(0)
            
            current_path_segment_idx = (current_path_segment_idx - 1) % len(segment_info)
            segment_info[current_path_segment_idx].set_progress(1)
        else:
            raise Exception("This shouldn't happen")


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
        if isinstance(data, (int, float, str)):
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