import numpy as np
from utils import \
    powerset \
    , get_observability_matrix as old_get_observability_matrix \
    , s_sparse_observability as old_get_s_sparse_observability

def np_make_mask(n, I):
    """
    Creates a mask of size n with 1s at the positions in I.
    """
    mask = np.full(n, False)
    mask[list(I)] = True
    return mask

def get_observability_matrix(A, C, N, S):
    """
    Calculates the observability matrix of a system with observability index N.
    Parameters:
        A: state transition matrix
        C: output matrix
        N: observability index (number of time steps)
        S: {0,1}^p vector indicating which sensors are available (1 means available, 0 means missing)
    """

    n = A.shape[0]
    p = C.shape[0]

    # Matrix that sets the attacked sensors to zero
    I_S = np.diag(S)

    O = np.zeros((p*N, n))
    for i in range(N):
        O[i*p:(i+1)*p, :] = I_S @ C @ np.linalg.matrix_power(A,i)
    
    return O

def get_s_sparse_observability(A,C,N,P):
    """
    Calculates the s-sparse observability of a system.
    Arguments:
        A: state transition matrix
        C: output matrix
        N: observability index (number of time steps)
        P: {0,1}^p vector indicating whether a sensor is protected (1 means protected, 0 means unprotected)
    """
    n = A.shape[0]
    p = C.shape[0]

    s = -1
    for K_tuple in powerset(range(p)):
        # K is a {0,1}^p vector indicating which sensors are missing
        K = np_make_mask(p, K_tuple)
        
        # Skip if a missing sensor is protected
        if np.any(np.logical_and(K, P)):
            continue

        # construct the sensor availability vector S
        S = np.ones(p)
        S[K] = 0

        O_K = get_observability_matrix(A, C, N, S)
        
        O_K_rank = np.linalg.matrix_rank(O_K)

        if O_K_rank < n:
            # found the first non-full-rank entry, which means
            # that the system no longer has full observability
            s = K.sum() - 1
            break

    if s == -1:
        # the system is fully observable
        s = p - P.sum()

    return s

def main():
    # test that get_observability_matrix is equivalent to old_get_observability_matrix
    A = np.array([[1, 1], [0, 1]])
    C = np.array([[1, 0], [0, 1]])
    N = A.shape[0]
    # available sensors
    # R = list(range(C.shape[0]))
    R = [0]
    # available sensors as a vector
    S = np.zeros(C.shape[0])
    S[R] = 1

    O = get_observability_matrix(A, C,  N, S)
    O_old = old_get_observability_matrix(A, C, R, N)
    # remove zero rows from O and compare it with O_old
    assert np.allclose(O[~np.all(O == 0, axis=1)], O_old)

    # test that get_s_sparse_observability is equivalent to old_get_s_sparse_observability
    A = np.array([[1, 1], [0, 1]])
    C = np.array([[1, 0], [0, 1]])
    N = A.shape[0]

    s = get_s_sparse_observability(A, C, N, np.zeros(C.shape[0]))
    s_old = old_get_s_sparse_observability(A, C)

    assert s == s_old

    # This system's s-sparse observability increases if we protect sensor 0
    A = np.array([[1, 1], [0, 1]])
    C = np.array([[1, 0], [0, 1]])
    N = A.shape[0]
    assert get_s_sparse_observability(A, C, N, np.array([0, 0])) == 0
    assert get_s_sparse_observability(A, C, N, np.array([1, 0])) == 1
    assert get_s_sparse_observability(A, C, N, np.array([0, 1])) == 0
    assert get_s_sparse_observability(A, C, N, np.array([1, 1])) == 0

    pass


if __name__ == "__main__":
    main()
