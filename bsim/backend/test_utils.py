import numpy as np
from numpy import sin, cos, tan

from utils import s_sparse_observability

def test_s_sparse_observability():
    # A 4x4 system
    A = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    n = A.shape[0]

    C = np.eye(n)
    assert s_sparse_observability(A, C) == 0

    C = np.concatenate((np.eye(n), np.eye(n)))
    assert s_sparse_observability(A, C) == 1

    C = np.concatenate((np.eye(n), np.eye(n), np.eye(n)))
    assert s_sparse_observability(A, C) == 2

    C = np.concatenate((np.eye(n), np.eye(n), np.eye(n), np.eye(n)))
    assert s_sparse_observability(A, C) == 3

    # Kinematic bicycle
    theta = np.pi / 4
    delta = np.pi / 8
    L = 2.9
    v = 5

    A = np.array([
        [0, 0, -v*sin(theta), cos(theta), 0],
        [0, 0, v*cos(theta), sin(theta), 0],
        [0, 0, 0, tan(delta)/L, v/(L*cos(theta) ** 2)],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])
    n = A.shape[0]

    C = np.eye(n)
    assert s_sparse_observability(A, C) == 0

    C = np.concatenate((np.eye(n), np.eye(n)))
    assert s_sparse_observability(A, C) == 1

    C = np.concatenate((np.eye(n), np.eye(n), np.eye(n)))
    assert s_sparse_observability(A, C) == 2

    C = np.concatenate((np.eye(n), np.eye(n), np.eye(n), np.eye(n)))
    assert s_sparse_observability(A, C) == 3

    # redundant x and y sensors
    C = np.array([
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ])
    assert s_sparse_observability(A, C) == 1

    # 3 pairs of x and y sensors
    C = np.array([
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ])
    assert s_sparse_observability(A, C) == 2

    # 4 pairs of x and y sensors
    C = np.array([
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ])
    assert s_sparse_observability(A, C) == 3

    # 5 pairs of x and y sensors
    C = np.array([
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ])
    assert s_sparse_observability(A, C) == 4


if __name__ == '__main__':
    test_s_sparse_observability()

    print("Tests passed!")