import control.matlab
import numpy as np
from numpy import sin, cos, tan

from utils import s_sparse_observability, get_evolution_matrices, get_observability_mapping, is_observable, is_attackable, is_observable_ltv, is_attackable_ltv, expand_sensor_configs_over_time, powerset

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

def test_get_evolution_matrices():
    assert all(
        np.array_equal(a1, a2) for a1, a2 in zip(
            get_evolution_matrices([np.eye(3)]*4, [np.eye(3)]*5),
            (
                np.concatenate([np.eye(3)]*5),
                np.concatenate([np.eye(3)]*5)
            )
        )
    )

    A = np.array([
        [1,2,3],
        [4,5,6],
        [7,8,9]
    ])
    C = np.array([
        [1,0,0],
        [0,1,0],
    ])
    assert all(
        np.array_equal(a1, a2) for a1, a2 in zip(
            get_evolution_matrices([A]*2, [C]*3),
            (
                np.concatenate([
                    np.eye(A.shape[0]),
                    A,
                    A@A
                ]),
                np.concatenate([
                    C,
                    C@A,
                    C@A@A
                ])
            )
        )
    )

def test_get_observability_mapping():
    C = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1],
    ])
    assert get_observability_mapping(C) == {
        0: [{0}],
        1: [{1}],
        2: [{2}],
    }

    C = np.array([
        [1],
        [1],
        [1],
    ])
    assert get_observability_mapping(C) == {
        0: [{0}, {1}, {2}]
    }

    C = np.array([
        [1,0],
        [0,1],
        [1,0],
    ])
    assert get_observability_mapping(C) == {
        0: [{0}, {2}],
        1: [{1}],
    }

    C = np.array([
        [1,0],
        [0,1],
        [1,1],
    ])
    assert get_observability_mapping(C) == {
        0: [{0}, {1, 2}],
        1: [{1}, {0, 2}],
    }

    C = np.array([
        [1,1],
        [1,-1],
    ])
    assert get_observability_mapping(C) == {
        0: [{0, 1}],
        1: [{0, 1}],
    }
    C = np.array([
        [1,0],
        [1,1],
        [1,-1],
    ])
    assert get_observability_mapping(C) == {
        0: [{0}, {1, 2}],
        1: [{0, 1}, {0, 2}, {1, 2}],
    }

    C = np.array([
        [1,0],
        [0,1],
        [1,1],
        [1,-1],
    ])
    assert get_observability_mapping(C) == {
        0: [{0}, {1, 2}, {1, 3}, {2, 3}],
        1: [{1}, {0, 2}, {0, 3}, {2, 3}],
    }

    C = np.array([
        [1,0],
        [0,1],
        [1,1],
        [1,-1],
        [1,2],
    ])
    assert get_observability_mapping(C) == {
        0: [{0}, {1, 2}, {1, 3}, {1, 4}, {2, 3}, {2, 4}, {3, 4}],
        1: [{1}, {0, 2}, {0, 3}, {0, 4}, {2, 3}, {2, 4}, {3, 4}],
    }

    # spice things up with an discrete LTI system
    A = np.array([
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
        [0,0,0,0],
    ])
    B = np.zeros((4,1))
    C = np.eye(4)
    D = 0
    
    sysd = control.matlab.c2d(control.matlab.ss(A,B,C,D), 0.1)

    assert get_observability_mapping(sysd.C) == {
        0: [{0}],
        1: [{1}],
        2: [{2}],
        3: [{3}],
    }
    assert get_observability_mapping(sysd.C@sysd.A) == {
        0: [{0,1,2,3}],
        1: [{1,2,3}],
        2: [{2,3}],
        3: [{3}],
    }

def test_is_observable():
    C = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1],
    ])
    mapping = get_observability_mapping(C)
    assert is_observable(mapping=mapping, missing_sensors=[]) == True
    assert is_observable(C=C, missing_sensors=[]) == True
    assert is_observable(mapping=mapping, missing_sensors=[0]) == False
    assert is_observable(mapping=mapping, missing_sensors=[1]) == False
    assert is_observable(mapping=mapping, missing_sensors=[2]) == False

    C = np.array([
        [1,0],
        [0,1],
        [1,1],
    ])
    assert is_observable(C=C, missing_sensors=[]) == True
    assert is_observable(C=C, missing_sensors=[0]) == True
    assert is_observable(C=C, missing_sensors=[1]) == True
    assert is_observable(C=C, missing_sensors=[2]) == True
    assert is_observable(C=C, missing_sensors=[0,1]) == False
    assert is_observable(C=C, missing_sensors=[0,2]) == False
    assert is_observable(C=C, missing_sensors=[1,2]) == False

    C = np.array([
        [1,0],
        [1,1],
        [1,-1],
    ])
    assert is_observable(C=C, missing_sensors=[]) == True
    assert is_observable(C=C, missing_sensors=[0]) == True
    assert is_observable(C=C, missing_sensors=[1]) == True
    assert is_observable(C=C, missing_sensors=[2]) == True
    assert is_observable(C=C, missing_sensors=[0,1]) == False
    assert is_observable(C=C, missing_sensors=[0,2]) == False
    assert is_observable(C=C, missing_sensors=[1,2]) == False

    C = np.array([
        [1,0],
        [0,1],
        [1,1],
        [1,-1],
    ])
    assert is_observable(C=C, missing_sensors=[]) == True
    assert is_observable(C=C, missing_sensors=[0]) == True
    assert is_observable(C=C, missing_sensors=[1]) == True
    assert is_observable(C=C, missing_sensors=[2]) == True
    assert is_observable(C=C, missing_sensors=[3]) == True
    assert is_observable(C=C, missing_sensors=[0,1]) == True
    assert is_observable(C=C, missing_sensors=[0,2]) == True
    assert is_observable(C=C, missing_sensors=[0,3]) == True
    assert is_observable(C=C, missing_sensors=[1,2]) == True
    assert is_observable(C=C, missing_sensors=[1,3]) == True
    assert is_observable(C=C, missing_sensors=[2,3]) == True
    assert is_observable(C=C, missing_sensors=[0,1,2]) == False
    assert is_observable(C=C, missing_sensors=[0,1,3]) == False
    assert is_observable(C=C, missing_sensors=[0,2,3]) == False
    assert is_observable(C=C, missing_sensors=[1,2,3]) == False

def test_is_attackable():
    C = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1],
    ])
    mapping = get_observability_mapping(C)
    assert is_attackable(mapping=mapping, attacked_sensors=[]) == True
    assert is_attackable(C=C, attacked_sensors=[]) == True
    assert is_attackable(mapping=mapping, attacked_sensors=[0]) == False
    assert is_attackable(mapping=mapping, attacked_sensors=[1]) == False
    assert is_attackable(mapping=mapping, attacked_sensors=[2]) == False

    C = np.array([
        [1,0],
        [0,1],
        [1,1],
    ])
    assert is_attackable(C=C, attacked_sensors=[]) == True
    assert is_attackable(C=C, attacked_sensors=[0]) == False
    assert is_attackable(C=C, attacked_sensors=[1]) == False
    assert is_attackable(C=C, attacked_sensors=[2]) == False
    assert is_attackable(C=C, attacked_sensors=[0,1]) == False
    assert is_attackable(C=C, attacked_sensors=[0,2]) == False
    assert is_attackable(C=C, attacked_sensors=[1,2]) == False

    C = np.array([
        [1,0],
        [0,1],
        [1,0],
    ])
    assert is_attackable(C=C, attacked_sensors=[]) == True
    assert is_attackable(C=C, attacked_sensors=[0]) == False
    assert is_attackable(C=C, attacked_sensors=[1]) == False
    assert is_attackable(C=C, attacked_sensors=[2]) == False
    assert is_attackable(C=C, attacked_sensors=[0,1]) == False
    assert is_attackable(C=C, attacked_sensors=[0,2]) == False
    assert is_attackable(C=C, attacked_sensors=[1,2]) == False

    C = np.array([
        [1,0],
        [0,1],
        [1,0],
        [1,0],
    ])
    assert is_attackable(C=C, attacked_sensors=[]) == True
    assert is_attackable(C=C, attacked_sensors=[0]) == True
    assert is_attackable(C=C, attacked_sensors=[1]) == False
    assert is_attackable(C=C, attacked_sensors=[2]) == True
    assert is_attackable(C=C, attacked_sensors=[3]) == True
    assert is_attackable(C=C, attacked_sensors=[0,1]) == False
    assert is_attackable(C=C, attacked_sensors=[0,2]) == False
    assert is_attackable(C=C, attacked_sensors=[0,3]) == False
    assert is_attackable(C=C, attacked_sensors=[1,2]) == False
    assert is_attackable(C=C, attacked_sensors=[1,3]) == False
    assert is_attackable(C=C, attacked_sensors=[2,3]) == False

    C = np.array([
        [1],
        [1],
        [1],
    ])
    assert is_attackable(C=C, attacked_sensors=[]) == True
    assert is_attackable(C=C, attacked_sensors=[0]) == True
    assert is_attackable(C=C, attacked_sensors=[1]) == True
    assert is_attackable(C=C, attacked_sensors=[2]) == True
    assert is_attackable(C=C, attacked_sensors=[0, 1]) == False
    assert is_attackable(C=C, attacked_sensors=[0, 2]) == False
    assert is_attackable(C=C, attacked_sensors=[1, 2]) == False

    C = np.array([
        [1,0],
        [0,1],
        [1,1],
        [1,-1],
    ])
    assert is_attackable(C=C, attacked_sensors=[]) == True
    # Attacking sensor 0 makes recovering state 1 ambiguous.
    # This is a strange dependency
    assert is_attackable(C=C, attacked_sensors=[0]) == False
    # Attacking sensor 0 makes recovering state 0 ambiguous.
    # This is a strange dependency
    assert is_attackable(C=C, attacked_sensors=[1]) == False
    assert is_attackable(C=C, attacked_sensors=[2]) == False
    assert is_attackable(C=C, attacked_sensors=[3]) == False

    C = np.array([
        [1,0],
        [0,1],
        [1,1],
        [1,-1],
        [1,2],
    ])
    assert is_attackable(C=C, attacked_sensors=[]) == True
    assert is_attackable(C=C, attacked_sensors=[0]) == True
    assert is_attackable(C=C, attacked_sensors=[1]) == True
    assert is_attackable(C=C, attacked_sensors=[2]) == True
    assert is_attackable(C=C, attacked_sensors=[3]) == True
    assert is_attackable(C=C, attacked_sensors=[0,1]) == False
    assert is_attackable(C=C, attacked_sensors=[0,2]) == False
    assert is_attackable(C=C, attacked_sensors=[0,3]) == False
    assert is_attackable(C=C, attacked_sensors=[0,4]) == False
    assert is_attackable(C=C, attacked_sensors=[1,2]) == False
    assert is_attackable(C=C, attacked_sensors=[1,3]) == False
    assert is_attackable(C=C, attacked_sensors=[1,4]) == False
    assert is_attackable(C=C, attacked_sensors=[2,3]) == False
    assert is_attackable(C=C, attacked_sensors=[2,4]) == False
    assert is_attackable(C=C, attacked_sensors=[3,4]) == False

def test_expand_sensor_configs_over_time():
    assert set(expand_sensor_configs_over_time([(0,1)], p=2, N=3)) == set([(0,1),(2,3),(4,5),(0,1,2,3,4,5)])
    assert set(expand_sensor_configs_over_time([(0,1),(2,3)], p=4, N=2)) == set([(0,1),(4,5),(2,3),(6,7),(0,1,4,5),(2,3,6,7)])
    assert set(expand_sensor_configs_over_time([(0,1),(2,3)], p=5, N=2)) == set([(0,1),(5,6),(2,3),(7,8),(0,1,5,6),(2,3,7,8)])
    assert set(expand_sensor_configs_over_time(powerset(range(2)), p=2, N=3)) == set([(),(0,),(1,),(0,1),(2,),(3,),(2,3),(4,),(5,),(4,5),(0,2,4),(1,3,5),(0,1,2,3,4,5)])

    # test that reducing the mapping size using sensor configuration expansion does not change the failure
    # domain of the mapping. Assuming sensors fail across all time steps.
    theta = np.pi / 4
    delta = np.pi / 8
    L = 2.9
    v = 5

    Ac = np.array([
        [0, 0, -v*sin(theta), cos(theta), 0],
        [0, 0, v*cos(theta), sin(theta), 0],
        [0, 0, 0, tan(delta)/L, v/(L*cos(theta) ** 2)],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])
    n = Ac.shape[0]
    Bc = np.zeros((n, 1))
    Cc = np.eye(n)
    Dc = 0

    sysd = control.matlab.c2d(control.matlab.ss(Ac, Bc, Cc, Dc), 0.1)
    A = sysd.A
    C = sysd.C
    p = C.shape[0]

    N = 2
    mapping_full = get_observability_mapping(get_evolution_matrices(Cs=[C]*N, As=[A]*(N-1))[1])
    mapping_reduced = get_observability_mapping(get_evolution_matrices(Cs=[C]*N, As=[A]*(N-1))[1], expand_sensor_configs_over_time(powerset(range(p)), p, N))
    # Check to see if the two mappings have equivalent observability properties
    for missing_sensors in powerset(range(p)):
        assert is_observable(mapping_full, missing_sensors) == is_observable(mapping_reduced, missing_sensors), "Failed for missing sensors: {}".format(missing_sensors)

    N = 3
    mapping_full = get_observability_mapping(get_evolution_matrices(Cs=[C]*N, As=[A]*(N-1))[1])
    mapping_reduced = get_observability_mapping(get_evolution_matrices(Cs=[C]*N, As=[A]*(N-1))[1], expand_sensor_configs_over_time(powerset(range(p)), p, N))
    # Check to see if the two mappings have equivalent observability properties
    for missing_sensors in powerset(range(p)):
        assert is_observable(mapping_full, missing_sensors) == is_observable(mapping_reduced, missing_sensors), "Failed for missing sensors: {}".format(missing_sensors)

    N = 4
    # WARNING: This line takes a long time
    mapping_full = get_observability_mapping(get_evolution_matrices(Cs=[C]*N, As=[A]*(N-1))[1])
    mapping_reduced = get_observability_mapping(get_evolution_matrices(Cs=[C]*N, As=[A]*(N-1))[1], expand_sensor_configs_over_time(powerset(range(p)), p, N))
    # Check to see if the two mappings have equivalent observability properties
    for missing_sensors in powerset(range(p)):
        assert is_observable(mapping_full, missing_sensors) == is_observable(mapping_reduced, missing_sensors), "Failed for missing sensors: {}".format(missing_sensors)

def test_is_observable_ltv():
    # A 4x4 system
    Ac = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    n = Ac.shape[0]
    Bc = np.zeros((n, 1))
    Cc = np.eye(n)
    Dc = 0

    sysd = control.matlab.c2d(control.matlab.ss(Ac, Bc, Cc, Dc), 0.1)
    A = sysd.A
    C = sysd.C

    assert is_observable_ltv(Cs=[C,C], As=[A], missing_sensors=[]) == True
    assert is_observable_ltv(Cs=[C,C], As=[A], missing_sensors=[0]) == False
    # when senor 1 is missing, we can use the change in sensor 0 to recover state 1
    assert is_observable_ltv(Cs=[C,C], As=[A], missing_sensors=[1]) == True
    assert is_observable_ltv(Cs=[C,C], As=[A], missing_sensors=[2]) == True
    assert is_observable_ltv(Cs=[C,C], As=[A], missing_sensors=[3]) == True
    assert is_observable_ltv(Cs=[C,C], As=[A], missing_sensors=[1,2]) == False
    assert is_observable_ltv(Cs=[C,C], As=[A], missing_sensors=[1,3]) == True
    assert is_observable_ltv(Cs=[C,C], As=[A], missing_sensors=[2,3]) == True

    # Kinematic bicycle
    theta = np.pi / 4
    delta = np.pi / 8
    L = 2.9
    v = 5

    Ac = np.array([
        [0, 0, -v*sin(theta), cos(theta), 0],
        [0, 0, v*cos(theta), sin(theta), 0],
        [0, 0, 0, tan(delta)/L, v/(L*cos(theta) ** 2)],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])
    n = Ac.shape[0]
    Bc = np.zeros((n, 1))
    Cc = np.eye(n)
    Dc = 0

    sysd = control.matlab.c2d(control.matlab.ss(Ac, Bc, Cc, Dc), 0.1)
    A = sysd.A
    C = sysd.C

    assert is_observable_ltv(Cs=[C]*n, As=[A]*(n-1), missing_sensors=[]) == True
    assert is_observable_ltv(Cs=[C]*n, As=[A]*(n-1), missing_sensors=[0]) == False
    assert is_observable_ltv(Cs=[C]*n, As=[A]*(n-1), missing_sensors=[1]) == False
    assert is_observable_ltv(Cs=[C]*n, As=[A]*(n-1), missing_sensors=[2]) == True
    assert is_observable_ltv(Cs=[C]*n, As=[A]*(n-1), missing_sensors=[3]) == True
    assert is_observable_ltv(Cs=[C]*n, As=[A]*(n-1), missing_sensors=[4]) == True
    assert is_observable_ltv(Cs=[C]*n, As=[A]*(n-1), missing_sensors=[2,3]) == True
    assert is_observable_ltv(Cs=[C]*n, As=[A]*(n-1), missing_sensors=[2,4]) == True
    assert is_observable_ltv(Cs=[C]*n, As=[A]*(n-1), missing_sensors=[3,4]) == True

    # Kinematic bicycle
    theta = np.pi / 4
    delta = np.pi / 8
    L = 2.9
    v = 5

    Ac = np.array([
        [0, 0, -v*sin(theta), cos(theta), 0],
        [0, 0, v*cos(theta), sin(theta), 0],
        [0, 0, 0, tan(delta)/L, v/(L*cos(theta) ** 2)],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])
    n = Ac.shape[0]
    Bc = np.zeros((n, 1))
    Cc = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        # redundancy on the x sensor
        [1, 0, 0, 0, 0],
    ])
    Dc = 0

    sysd = control.matlab.c2d(control.matlab.ss(Ac, Bc, Cc, Dc), 0.1)
    A = sysd.A
    C = sysd.C

    assert is_observable_ltv(Cs=[C]*n, As=[A]*(n-1), missing_sensors=[]) == True
    assert is_observable_ltv(Cs=[C]*n, As=[A]*(n-1), missing_sensors=[0]) == True
    assert is_observable_ltv(Cs=[C]*n, As=[A]*(n-1), missing_sensors=[1]) == False
    assert is_observable_ltv(Cs=[C]*n, As=[A]*(n-1), missing_sensors=[2]) == True
    assert is_observable_ltv(Cs=[C]*n, As=[A]*(n-1), missing_sensors=[3]) == True
    assert is_observable_ltv(Cs=[C]*n, As=[A]*(n-1), missing_sensors=[4]) == True
    assert is_observable_ltv(Cs=[C]*n, As=[A]*(n-1), missing_sensors=[5]) == True
    assert is_observable_ltv(Cs=[C]*n, As=[A]*(n-1), missing_sensors=[0,1]) == False
    assert is_observable_ltv(Cs=[C]*n, As=[A]*(n-1), missing_sensors=[0,2]) == True
    assert is_observable_ltv(Cs=[C]*n, As=[A]*(n-1), missing_sensors=[0,3]) == True
    assert is_observable_ltv(Cs=[C]*n, As=[A]*(n-1), missing_sensors=[0,4]) == True
    assert is_observable_ltv(Cs=[C]*n, As=[A]*(n-1), missing_sensors=[0,5]) == False
    assert is_observable_ltv(Cs=[C]*n, As=[A]*(n-1), missing_sensors=[2,3]) == True
    assert is_observable_ltv(Cs=[C]*n, As=[A]*(n-1), missing_sensors=[2,4]) == True
    assert is_observable_ltv(Cs=[C]*n, As=[A]*(n-1), missing_sensors=[3,4]) == True


def test_ltv_and_s_sparse_observability_equivalence():
    def get_ltv_s_sparse_observability(As,Cs):
        """
        Given an LTV system, returns the s-sparse observability
        """
        for missing_sensors in powerset(range(Cs[0].shape[0])):
            if not is_observable_ltv(Cs=Cs, As=As, missing_sensors=missing_sensors):
                return len(missing_sensors) - 1
        
        return Cs[0].shape[0]

    # Kinematic bicycle
    theta = np.pi / 4
    delta = np.pi / 8
    L = 2.9
    v = 5

    Ac = np.array([
        [0, 0, -v*sin(theta), cos(theta), 0],
        [0, 0, v*cos(theta), sin(theta), 0],
        [0, 0, 0, tan(delta)/L, v/(L*cos(theta) ** 2)],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])
    n = Ac.shape[0]
    Bc = np.zeros((n, 1))
    Cc = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        # redundancy on the x sensor
        [1, 0, 0, 0, 0],
    ])
    Dc = 0

    sysd = control.matlab.c2d(control.matlab.ss(Ac, Bc, Cc, Dc), 0.1)
    A = sysd.A
    C = sysd.C

    assert s_sparse_observability(A, C) == get_ltv_s_sparse_observability([A]*(n-1), [C]*n)

    A = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])
    n = A.shape[0]

    C = np.eye(n)
    assert s_sparse_observability(A, C) == get_ltv_s_sparse_observability([A]*(n-1), [C]*n)

    C = np.concatenate((np.eye(n), np.eye(n)))
    assert s_sparse_observability(A, C) == get_ltv_s_sparse_observability([A]*(n-1), [C]*n)

if __name__ == '__main__':
    test_s_sparse_observability()
    test_get_evolution_matrices()
    test_get_observability_mapping()
    test_is_observable()
    test_is_attackable()
    test_expand_sensor_configs_over_time()
    test_is_observable_ltv()
    # test_is_attackable_ltv()
    test_ltv_and_s_sparse_observability_equivalence()

    print("Tests passed!")