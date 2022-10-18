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


def calc_input_effects_on_output(A, B, C, inputs):
    # calculates the input effects on the output
    # A: numpy.ndarray - matrix of size (n,n)
    # B: numpy.ndarray - matrix of size (n,m)
    # C: numpy.ndarray - matrix of size (p,n)
    # inputs: numpy.ndarray - matrix of size (m,T)
    # returns: numpy.ndarray - matrix of size (p,T)

    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]
    T = inputs.shape[1]

    assert A.shape == (n, n)
    assert B.shape == (n, m)
    assert C.shape == (p, n)
    assert inputs.shape == (m, T)

    # TODO: make this faster by reusing the calculations for Phi
    effects = np.zeros((p, T))
    for k in range(T):
        for j in range(k):
            effects[:, k] += C @ matrix_power(A, k-1-j) @ B @ inputs[:, j]

    return effects


def optimize_l1(n, p, T, Phi, Y):
    # solves the l1/l2 norm minimization problem
    # n: int - number of states
    # p: int - number of outputs
    # T: int - number of time steps
    # Phi: numpy.ndarray - matrix of size (p*T, n) - (C*A^0, C*A^1, ..., C*A^(T-1))'
    # Y: numpy.ndarray - measured outputs, with input effects subtracted, size (p*T)
    # returns: numpy.ndarray

    assert Phi.shape == (p*T, n)
    assert Y.shape == (p*T,)

    x0_hat = cp.Variable(n)
    # define the expression that we want to run l1/l2 optimization on
    optimizer = Y - np.matmul(Phi, x0_hat)
    # reshape to adapt to l1/l2 norm formulation
    # Note that cp uses fortran ordering (column-major), this is different from numpy,
    # which uses c ordering (row-major)
    optimizer_reshaped = cp.reshape(optimizer, (p, T))
    optimizer_final = cp.mixed_norm(optimizer_reshaped, p=2, q=1)
    # Equivalent to optimizer_final = cp.norm1(cp.norm(optimizer_reshaped, axis=1))

    obj = cp.Minimize(optimizer_final)

    # Form and solve problem.
    prob = cp.Problem(obj)
    prob.solve(verbose=True)  # Returns the optimal value.

    return (prob, x0_hat)


def optimize_l0(n, p, T, Phi, Y, eps=0.2):
    # solves the l0 minimization problem
    # n: int - number of states
    # p: int - number of outputs
    # T: int - number of time steps
    # Phi: numpy.ndarray - matrix of size (p*T, n) - (C*A^0, C*A^1, ..., C*A^(T-1))'
    # Y: numpy.ndarray - measured outputs, with input effects subtracted, size (p*T)
    # returns: numpy.ndarray

    assert Phi.shape == (p*T, n)
    assert Y.shape == (p*T,)

    # with Pool(processes=30) as pool:
    #     solns = pool.starmap(optimize_l0_subproblem, [(n, p, T, Phi, Y, attacked_sensor_indices) for attacked_sensor_indices in powerset(range(p))])
    for attacked_sensor_indices in powerset(range(p)):
        prob, x_hat_l0 = optimize_l0_subproblem(n, p, T, Phi, Y, attacked_sensor_indices, eps)
        if prob.status in ["optimal", "optimal_inaccurate"]:
            return (prob, x_hat_l0)

    raise Exception("No solution found")


def optimize_l0_subproblem(n, p, T, Phi, Y, attacked_sensor_indices, eps):
    x0_hat = cp.Variable(n)
    optimizer = Y - np.matmul(Phi, x0_hat)
    optimizer_reshaped = cp.reshape(optimizer, (p, T))
    optimizer_final = cp.mixed_norm(optimizer_reshaped, p=2, q=1)

    # Support both scalar eps and eps per sensor
    if np.isscalar(eps):
        eps = np.ones(p) * eps

    constraints = []
    for j in set(range(p)) - set(attacked_sensor_indices):
        for t in range(T):
            # constraints.append(cp.norm(optimizer[p*t+j]) <= eps)
            constraints.append(optimizer[p*t+j] <= eps[j])
            constraints.append(optimizer[p*t+j] >= -eps[j])
    
    prob = cp.Problem(cp.Minimize(cp.norm(optimizer_final)), constraints)
    start = time.time()
    prob.solve()
    end = time.time()

    print(f"Solved l0 subproblem in {end-start:.2f} seconds. Indices: {attacked_sensor_indices}, status: {prob.status}, value: {prob.value}")

    return (prob, x0_hat)


def get_l0_state_estimation_l2_bound(A: np.ndarray, C: np.ndarray, largest_sensor_deviations: np.ndarray, qmax: int, T: int):
    """
    Calculates the l0 state estimation bound for a given system and sensor deviations.
    """
    n = A.shape[0]
    p = C.shape[0]

    Rs = [list(s) for s in powerset(range(p)) if len(s) == p - 2*qmax]

    bound = -np.inf

    for R in Rs:
        max_noise_norm = np.linalg.norm(largest_sensor_deviations[R])

        # C with only sensors in R
        PC = C[R, :]

        # Calculate O_R
        O_R = np.zeros((len(R)*T, n))
        for t in range(T):
            O_R[t*len(R):(t+1)*len(R), :] = PC @ matrix_power(A, t)

        pinv_O_R = np.linalg.pinv(O_R)

        largest_singular_value = max(np.linalg.svd(pinv_O_R, compute_uv=False))

        bound = max(bound, largest_singular_value * 2*max_noise_norm)

    return bound

def matrix_spectral_norm(A):
    return max(np.linalg.eigvals(A))

def get_error_estimation_l2_bounds(A: np.ndarray, C: np.ndarray, Dx: float, largest_sensor_deviations: np.ndarray, T: int):
    """
    Calculates the error estimation bounds for a given system and sensor deviations.
    """
    n = A.shape[0]
    p = C.shape[0]

    bounds = np.zeros(p)

    for i in range(p):
        O_i = np.zeros((T, n))
        for t in range(T):
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
        PC = C[remaining_sensors, :] # C with only sensors in K^c
        p_Kc = len(remaining_sensors)

        # Calculate observability matrix
        O_K = np.zeros((p*n, n))
        for i in range(n):
            O_K[i*p_Kc:(i+1)*p_Kc, :] = PC @ matrix_power(A, i)
        
        O_K_rank = np.linalg.matrix_rank(O_K)

        if s == -1 and O_K_rank < n:
            # found the first non-full-rank entry
            s = len(K) - 1
            break

    return s


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