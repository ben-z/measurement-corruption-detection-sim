import numpy as np
from utils import powerset, get_observability_matrix as old_get_observability_matrix

def get_observability_matrix(A, C, N):
    """
    Calculates the observability matrix of a system with observability index N.
    """

    n = A.shape[0]
    p = C.shape[0]

    O = np.zeros((p*N, n))
    for i in range(N):
        O[i*p:(i+1)*p, :] = C @ np.linalg.matrix_power(A,i)
    
    return O

def main():
    # test that get_observability_matrix is equivalent to old_get_observability_matrix
    A = np.array([[1, 1], [0, 1]])
    C = np.array([[1, 0], [0, 1]])
    N = 2
    O = get_observability_matrix(A, C, N)
    O_old = old_get_observability_matrix(A, C, None, N)
    assert np.allclose(O, O_old)

if __name__ == "__main__":
    main()
