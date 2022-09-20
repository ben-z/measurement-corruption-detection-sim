from numpy.linalg import matrix_power
import cvxpy as cp
import numpy as np
import json

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
            effects[:, k] += np.matmul(np.matmul(C, matrix_power(A, k-1-j)),
                                       np.matmul(B, inputs[:, j]))

    return effects


def optimize_l1(n, p, T, Phi, y):
    # solves the l1/l2 norm minimization problem
    # n: int - number of states
    # p: int - number of outputs
    # T: int - number of time steps
    # Phi: numpy.ndarray - matrix of size (p*T, n) - (C*A^0, C*A^1, ..., C*A^(T-1))'
    # y: numpy.ndarray - measured outputs, with input effects subtracted, size (p*T)
    # returns: numpy.ndarray

    assert Phi.shape == (p*T, n)
    assert y.shape == (p*T,)

    x0_hat_l1 = cp.Variable(n)
    # define the expression that we want to run l1/l2 optimization on
    optimizer = y - np.matmul(Phi, x0_hat_l1)
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

    return (prob, x0_hat_l1)


class JSONNumpyDecoder(json.JSONDecoder):
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