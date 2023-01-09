import control.matlab
import numpy as np
from numpy import sin, cos, tan

from utils import s_sparse_observability, get_evolution_matrices, get_observability_mapping, is_observable, is_attackable, is_observable_ltv, is_attackable_ltv

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
        0: set([(0,)]),
        1: set([(1,)]),
        2: set([(2,)]),
    }

    C = np.array([
        [1],
        [1],
        [1],
    ])
    assert get_observability_mapping(C) == {
        0: {(0,), (1,), (2,)}
    }

    C = np.array([
        [1,0],
        [0,1],
        [1,0],
    ])
    assert get_observability_mapping(C) == {
        0: set([(0,), (2,)]),
        1: set([(1,)]),
    }

    C = np.array([
        [1,0],
        [0,1],
        [1,1],
    ])
    assert get_observability_mapping(C) == {
        0: set([(0,), (1, 2)]),
        1: set([(1,), (0, 2)]),
    }

    C = np.array([
        [1,1],
        [1,-1],
    ])
    assert get_observability_mapping(C) == {
        0: set([(0, 1)]),
        1: set([(0, 1)]),
    }
    C = np.array([
        [1,0],
        [1,1],
        [1,-1],
    ])
    assert get_observability_mapping(C) == {
        0: set([(0,), (1, 2)]),
        1: set([(0, 1), (0, 2), (1, 2)]),
    }

    C = np.array([
        [1,0],
        [0,1],
        [1,1],
        [1,-1],
    ])
    assert get_observability_mapping(C) == {
        0: set([(0,), (1, 2), (1, 3), (2, 3)]),
        1: set([(1,), (0, 2), (0, 3), (2, 3)]),
    }

    C = np.array([
        [1,0],
        [0,1],
        [1,1],
        [1,-1],
        [1,2],
    ])
    assert get_observability_mapping(C) == {
        0: set([(0,), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]),
        1: set([(1,), (0, 2), (0, 3), (0, 4), (2, 3), (2, 4), (3, 4)]),
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
        0: set([(0,)]),
        1: set([(1,)]),
        2: set([(2,)]),
        3: set([(3,)]),
    }
    assert get_observability_mapping(sysd.C@sysd.A) == {
        0: set([(0,1,2,3)]),
        1: set([(1,2,3)]),
        2: set([(2,3)]),
        3: set([(3,)]),
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

    assert is_observable_ltv(Cs=[C,C], As=[A], missing_sensors=[]) == True
    assert is_observable_ltv(Cs=[C,C], As=[A], missing_sensors=[0]) == False
    assert is_observable_ltv(Cs=[C,C], As=[A], missing_sensors=[1]) == False
    assert is_observable_ltv(Cs=[C,C], As=[A], missing_sensors=[2]) == True
    assert is_observable_ltv(Cs=[C,C], As=[A], missing_sensors=[3]) == True
    assert is_observable_ltv(Cs=[C,C], As=[A], missing_sensors=[4]) == True
    assert is_observable_ltv(Cs=[C,C], As=[A], missing_sensors=[2,3]) == True
    assert is_observable_ltv(Cs=[C,C], As=[A], missing_sensors=[2,4]) == False
    assert is_observable_ltv(Cs=[C,C], As=[A], missing_sensors=[3,4]) == True

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

    assert is_observable_ltv(Cs=[C]*3, As=[A]*(3-1), missing_sensors=[]) == True
    assert is_observable_ltv(Cs=[C]*3, As=[A]*(3-1), missing_sensors=[0]) == True
    assert is_observable_ltv(Cs=[C]*3, As=[A]*(3-1), missing_sensors=[1]) == False
    assert is_observable_ltv(Cs=[C]*3, As=[A]*(3-1), missing_sensors=[2]) == True
    assert is_observable_ltv(Cs=[C]*3, As=[A]*(3-1), missing_sensors=[3]) == True
    assert is_observable_ltv(Cs=[C]*3, As=[A]*(3-1), missing_sensors=[4]) == True
    assert is_observable_ltv(Cs=[C]*3, As=[A]*(3-1), missing_sensors=[5]) == True
    assert is_observable_ltv(Cs=[C]*3, As=[A]*(3-1), missing_sensors=[0,1]) == False
    assert is_observable_ltv(Cs=[C]*3, As=[A]*(3-1), missing_sensors=[0,2]) == True
    assert is_observable_ltv(Cs=[C]*3, As=[A]*(3-1), missing_sensors=[0,3]) == True
    assert is_observable_ltv(Cs=[C]*3, As=[A]*(3-1), missing_sensors=[0,4]) == True
    assert is_observable_ltv(Cs=[C]*3, As=[A]*(3-1), missing_sensors=[0,5]) == False
    assert is_observable_ltv(Cs=[C]*3, As=[A]*(3-1), missing_sensors=[2,3]) == True
    assert is_observable_ltv(Cs=[C]*3, As=[A]*(3-1), missing_sensors=[2,4]) == True
    assert is_observable_ltv(Cs=[C]*3, As=[A]*(3-1), missing_sensors=[3,4]) == True


if __name__ == '__main__':
    test_s_sparse_observability()
    test_get_evolution_matrices()
    test_get_observability_mapping()
    test_is_observable()
    test_is_attackable()

    print("Tests passed!")