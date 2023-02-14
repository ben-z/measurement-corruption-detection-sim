import numpy as np
from numpy import sin, cos, tan
from itertools import combinations
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

def toBinVec(S, n):
    """
    Converts a set S to a binary vector of length n.
    """
    return np_make_mask(n, S)

def toSet(binVec):
    """
    Converts a binary vector to a set.
    """
    return frozenset(np.where(binVec)[0])

def get_observability_matrix(A, C, N, S):
    """
    Calculates the observability matrix of a system with observability index N.
    Parameters:
        A: state transition matrix
        C: output matrix
        N: observability index (the number of time steps)
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
        N: observability index (the number of time steps)
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

def is_observable(A, C, N, S):
    """
    Checks if a system is observable.
    Parameters:
        A: state transition matrix
        C: output matrix
        N: observability index (the number of time steps)
        S: {0,1}^p vector indicating which sensors are available (1 means available, 0 means missing)
    """
    O = get_observability_matrix(A, C, N, S)
    return np.linalg.matrix_rank(O) == A.shape[0]

def get_s_sparse_observability2(A,C,N,P):
    """
    Calculates the s-sparse observability of a system.
    Arguments:
        A: state transition matrix
        C: output matrix
        N: observability index (the number of time steps)
        P: {0,1}^p vector indicating whether a sensor is protected (1 means protected, 0 means unprotected)
    Note: this is the same as get_s_sparse_observability, but it uses a different method
    """

    n = A.shape[0]
    p = C.shape[0]

    unprotected_sensors = np.where(P == 0)[0]

    # In each iteration, check if the system is observable when s_candidate sensors are missing.
    # If the system is observable, then the system has s_candidate-sparse observability.
    # If the system is not observable, then we return the largest s_candidate found.
    for s_candidate in range(1, len(unprotected_sensors)+1):
        # all combinations of s_candidate sensors that can go missing
        Ks = np.array([np_make_mask(p, setK) for setK in combinations(unprotected_sensors, s_candidate)])

        observable_mask = np.array([is_observable(A, C, N, ~K) for K in Ks])

        if not np.all(observable_mask):
            # found the first non-full-rank entry, which means
            # that the system no longer has full observability
            s = s_candidate - 1
            # sensors that when removed, the system is no longer observable
            important_Ks = Ks[~observable_mask]
            # sensors that when removed, the system is still observable
            unimportant_Ks = Ks[observable_mask]
            break
    else:
        # the system is fully sparse-observable, i.e. any unprotected sensor can be removed
        # without affecting the observability of the system
        s = len(unprotected_sensors)
        important_Ks = np.array([])
        unimportant_Ks = np.array([])
    
    return s, important_Ks, unimportant_Ks

def dev_tests():
    # Tests written during development (can be incomplete and messy)

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

    # Test that get_s_sparse_observability2 is equivalent to get_s_sparse_observability
    A = np.array([[1, 1], [0, 1]])
    C = np.array([[1, 0], [0, 1]])
    N = A.shape[0]
    P = np.array([0, 0])
    s, important_Ks, unimportant_Ks = get_s_sparse_observability2(A, C, N, P)
    s_old = get_s_sparse_observability(A, C, N, P)
    assert s == s_old

    # Test that get_s_sparse_observability2 is equivalent to get_s_sparse_observability
    A = np.array([[1, 1], [0, 1]])
    C = np.array([[1, 0], [0, 1]])
    N = A.shape[0]
    P = np.array([1, 0])
    s, important_Ks, unimportant_Ks = get_s_sparse_observability2(A, C, N, P)
    s_old = get_s_sparse_observability(A, C, N, P)
    assert s == s_old

    # Test that get_s_sparse_observability2 is equivalent to get_s_sparse_observability
    A = np.array([[1, 1], [0, 1]])
    C = np.array([[1, 0], [0, 1]])
    N = A.shape[0]
    P = np.array([0, 1])
    s, important_Ks, unimportant_Ks = get_s_sparse_observability2(A, C, N, P)
    s_old = get_s_sparse_observability(A, C, N, P)
    assert s == s_old

    # Test that get_s_sparse_observability2 is equivalent to get_s_sparse_observability
    A = np.array([[1, 1], [0, 1]])
    C = np.array([[1, 0], [0, 1]])
    N = A.shape[0]
    P = np.array([1, 1])
    s, important_Ks, unimportant_Ks = get_s_sparse_observability2(A, C, N, P)
    s_old = get_s_sparse_observability(A, C, N, P)
    assert s == s_old

    pass

def test_s_sparse_observability2():
    A = np.array([[1, 1], [0, 1]])
    C = np.array([[1, 0], [0, 1]])
    # observability index, or the number of time steps
    N = A.shape[0]

    # protected sensors
    P = toBinVec({}, C.shape[0])
    s, important_Ks, unimportant_Ks = get_s_sparse_observability2(A, C, N, P)
    assert s == 0
    assert set(toSet(K) for K in important_Ks) == set([frozenset({0})])
    assert set(toSet(K) for K in unimportant_Ks) == set([frozenset({1})])

    P = toBinVec({0}, C.shape[0])
    s, important_Ks, unimportant_Ks = get_s_sparse_observability2(A, C, N, P)
    assert s == 1
    assert set(toSet(K) for K in important_Ks) == set()
    assert set(toSet(K) for K in unimportant_Ks) == set()

    P = toBinVec({1}, C.shape[0])
    s, important_Ks, unimportant_Ks = get_s_sparse_observability2(A, C, N, P)
    assert s == 0
    assert set(toSet(K) for K in important_Ks) == set([frozenset({0})])
    assert set(toSet(K) for K in unimportant_Ks) == set()

    P = toBinVec({0,1}, C.shape[0])
    s, important_Ks, unimportant_Ks = get_s_sparse_observability2(A, C, N, P)
    assert s == 0
    assert set(toSet(K) for K in important_Ks) == set()
    assert set(toSet(K) for K in unimportant_Ks) == set()

    # Tests using a kinematic bicycle model
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
    # observability index, or the number of time steps
    N = A.shape[0]

    # Identity matrix
    C = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ])
    P = toBinVec({}, C.shape[0])
    s, important_Ks, unimportant_Ks = get_s_sparse_observability2(A, C, N, P)
    assert s == 0
    assert set(toSet(K) for K in important_Ks) == set([frozenset({0}), frozenset({1})])
    assert set(toSet(K) for K in unimportant_Ks) == set([frozenset({2}), frozenset({3}), frozenset({4})])
    P = toBinVec({0}, C.shape[0])
    s, important_Ks, unimportant_Ks = get_s_sparse_observability2(A, C, N, P)
    assert s == 0
    assert set(toSet(K) for K in important_Ks) == set([frozenset({1})])
    assert set(toSet(K) for K in unimportant_Ks) == set([frozenset({2}), frozenset({3}), frozenset({4})])
    P = toBinVec({0,1}, C.shape[0])
    s, important_Ks, unimportant_Ks = get_s_sparse_observability2(A, C, N, P)
    assert s == 3 # full s-sparse observability!
    assert set(toSet(K) for K in important_Ks) == set()
    assert set(toSet(K) for K in unimportant_Ks) == set()

    # Redundancy on x
    C = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0],
    ])
    P = toBinVec({}, C.shape[0])
    s, important_Ks, unimportant_Ks = get_s_sparse_observability2(A, C, N, P)
    assert s == 0
    assert set(toSet(K) for K in important_Ks) == set([frozenset({1})])
    assert set(toSet(K) for K in unimportant_Ks) == set([frozenset({0}), frozenset({2}), frozenset({3}), frozenset({4}), frozenset({5})])

    # Redundancy on x and y
    C = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
    ])
    P = toBinVec({}, C.shape[0])
    s, important_Ks, unimportant_Ks = get_s_sparse_observability2(A, C, N, P)
    assert s == 1
    assert set(toSet(K) for K in important_Ks) == set([frozenset({0,5}), frozenset({1,6})])
    assert set(toSet(K) for K in unimportant_Ks) == set([
        frozenset({0, 1}), frozenset({0, 2}), frozenset({0, 3}), frozenset({0, 4}), frozenset({0, 6}),
        frozenset({1, 2}), frozenset({1, 3}), frozenset({1, 4}), frozenset({1, 5}),
        frozenset({2, 3}), frozenset({2, 4}), frozenset({2, 5}), frozenset({2, 6}),
        frozenset({3, 4}), frozenset({3, 5}), frozenset({3, 6}),
        frozenset({4, 5}), frozenset({4, 6}),
        frozenset({5, 6}),
    ])

    pass

def main():
    # dev_tests()
    test_s_sparse_observability2()




if __name__ == "__main__":
    main()
    print("All tests passed!")
