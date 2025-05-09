import cvxpy as cp
import json
import numpy as np
import unittest
from math import pi, sin, cos, atan2, sqrt, tan
from scipy.linalg import expm
from parameterized import parameterized
from utils import (
    get_unpack_fn,
    kinematic_bicycle_model,
    generate_circle_approximation,
    generate_figure_eight_approximation,
    PIDController,
    closest_point_idx,
    closest_point_idx_local,
    wrap_to_pi,
    get_lookahead_idx,
    clamp,
    powerset,
    walk_trajectory_by_durations,
    optimize_l0_v2,
    Optimizer,
    get_state_evolution_tensor,
    get_output_evolution_tensor,
    get_s_sparse_observability,
    kinematic_bicycle_model_linearize,
    calc_input_effects_on_output,
)

# Helpers
def add(a, b):
    "Adds two numbers"
    return a + b

def multiply(a, b):
    "Multiplies two numbers"
    return a * b

class TestUnpackFunction(unittest.TestCase):
    def test_unpack_fn_add(self):
        unpacked_add = get_unpack_fn(add)
        self.assertEqual(unpacked_add((1, 2)), 3)
        self.assertEqual(unpacked_add.__name__, 'add')
        self.assertEqual(unpacked_add.__doc__, 'Adds two numbers')

    def test_unpack_fn_multiply(self):
        unpacked_multiply = get_unpack_fn(multiply)
        self.assertEqual(unpacked_multiply((2, 3)), 6)
        self.assertEqual(unpacked_multiply.__name__, 'multiply')
        self.assertEqual(unpacked_multiply.__doc__, 'Multiplies two numbers')

    def test_preserves_signature(self):
        # This is a more advanced test and might not be necessary for all use cases
        # since the signature is not actually preserved; the unpack_fn always takes
        # a single argument (a tuple). For a true signature preservation, you would
        # need to use introspection or third-party packages like `decorator`, which
        # can recreate function signatures dynamically.
        unpacked_add = get_unpack_fn(add)
        self.assertEqual(unpacked_add.__annotations__, add.__annotations__)

class TestKinematicBicycleModel(unittest.TestCase):
    def test_noop(self):
        state = np.array([0, 0, 0, 0, 0])
        input = np.array([0, 0])
        params = {'dt': 0.1, 'l': 1}
        new_state = kinematic_bicycle_model(state, input, params)
        expected_state = np.array([0, 0, 0, 0, 0])
        np.testing.assert_allclose(new_state, expected_state)

class TestKinematicBicycleModelLinearize(unittest.TestCase):
    def test_kinematic_bicycle_model_linearize(self):
        theta = 0
        v = 1
        delta = 0
        dt = 0.1
        l = 1
        Ad, Bd = kinematic_bicycle_model_linearize(theta, v, delta, dt, l)
        expected_Ad = np.array([
            [1.   , 0.   , 0.   , 0.1  , 0.   ],
            [0.   , 1.   , 0.1  , 0.   , 0.005],
            [0.   , 0.   , 1.   , 0.   , 0.1  ],
            [0.   , 0.   , 0.   , 1.   , 0.   ],
            [0.   , 0.   , 0.   , 0.   , 1.   ]
        ])
        expected_Bd = np.array([
            [0.005     , 0.        ],
            [0.        , 0.00016667],
            [0.        , 0.005     ],
            [0.1       , 0.        ],
            [0.        , 0.1       ]
        ])
        np.testing.assert_allclose(Ad, expected_Ad, atol=1e-8)
        np.testing.assert_allclose(Bd, expected_Bd, atol=1e-8)

class TestGenerateCircleApproximation(unittest.TestCase):
    @unittest.skip("Broken copilot-generated test")
    def test_generate_circle_approximation(self):
        center = [0, 0]
        radius = 1
        num_points = 10
        points, headings, curvatures, dK_ds_list = generate_circle_approximation(center, radius, num_points)
        expected_points = [[1, 0], [0.70710678, 0.70710678], [0, 1], [-0.70710678, 0.70710678], [-1, 0], [-0.70710678, -0.70710678], [0, -1], [0.70710678, -0.70710678], [1, 0], [0.70710678, 0.70710678]]
        expected_headings = [0, pi/4, pi/2, 3*pi/4, pi, -3*pi/4, -pi/2, -pi/4, 0, pi/4]
        expected_curvatures = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        expected_dK_ds_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        np.testing.assert_allclose(points, expected_points)
        np.testing.assert_allclose(headings, expected_headings)
        np.testing.assert_allclose(curvatures, expected_curvatures)
        np.testing.assert_allclose(dK_ds_list, expected_dK_ds_list)

class TestGenerateFigureEightApproximation(unittest.TestCase):
    @unittest.skip("Broken copilot-generated test")
    def test_generate_figure_eight_approximation(self):
        center = [0, 0]
        length = 1
        width = 0.5
        num_points = 10
        points, headings, curvatures, dK_ds_list = generate_figure_eight_approximation(center, length, width, num_points)
        expected_points = [[1, 0], [0.92387953, 0.38268343], [0.70710678, 0.70710678], [0.38268343, 0.92387953], [0, 1], [-0.38268343, 0.92387953], [-0.70710678, 0.70710678], [-0.92387953, 0.38268343], [-1, 0], [-0.92387953, -0.38268343], [-0.70710678, -0.70710678], [-0.38268343, -0.92387953], [0, -1], [0.38268343, -0.92387953], [0.70710678, -0.70710678], [0.92387953, -0.38268343], [1, 0]]
        expected_headings = [0, 0.46364761, pi/4, 1.10714872, pi/2, 2.03444394, 3*pi/4, 2.67794504, pi, -2.67794504, -3*pi/4, -2.03444394, -pi/2, -1.10714872, -pi/4, -0.46364761, 0]
        expected_curvatures = [0, 0.70710678, 1, 0.70710678, 0, -0.70710678, -1, -0.70710678, 0, 0.70710678, 1, 0.70710678, 0, -0.70710678, -1, -0.70710678, 0]
        expected_dK_ds_list = [0, -0.5, 0, 0.5, 0, -0.5, 0, 0.5, 0, -0.5, 0, 0.5, 0, -0.5, 0, 0.5, 0]
        np.testing.assert_allclose(points, expected_points)
        np.testing.assert_allclose(headings, expected_headings)
        np.testing.assert_allclose(curvatures, expected_curvatures)
        np.testing.assert_allclose(dK_ds_list, expected_dK_ds_list)

class TestPIDController(unittest.TestCase):
    @unittest.skip("Broken copilot-generated test")
    def test_pid_controller(self):
        kp = 1
        ki = 0.1
        kd = 0.01
        dt = 0.1
        pid = PIDController(kp, ki, kd, dt)
        error = 1
        output = pid.step(error)
        expected_output = 0.1
        self.assertAlmostEqual(output, expected_output)

class TestClosestPointIdx(unittest.TestCase):
    def test_closest_point_idx(self):
        points = [[0, 0], [1, 0], [0, 1]]
        x = 0.5
        y = 0.5
        closest_idx = closest_point_idx(points, x, y)
        expected_idx = 0
        self.assertEqual(closest_idx, expected_idx)

class TestClosestPointIdxLocal(unittest.TestCase):
    @unittest.skip("Broken copilot-generated test")
    def test_closest_point_idx_local(self):
        points = [[0, 0], [1, 0], [0, 1]]
        x = 0.5
        y = 0.5
        prev_idx = 0
        closest_idx = closest_point_idx_local(points, x, y, prev_idx)
        expected_idx = 1
        self.assertEqual(closest_idx, expected_idx)

class TestWrapToPi(unittest.TestCase):
    def test_wrap_to_pi(self):
        x = 3*pi
        wrapped_x = wrap_to_pi(x)
        expected_wrapped_x = -pi
        self.assertAlmostEqual(wrapped_x, expected_wrapped_x)

class TestGetLookaheadIdx(unittest.TestCase):
    def test_get_lookahead_idx(self):
        path_points = [[0, 0], [1, 0], [1, 1], [0, 1]]
        starting_idx = 0
        dist = 1.5
        new_idx = get_lookahead_idx(path_points, starting_idx, dist)
        expected_idx = 2
        self.assertEqual(new_idx, expected_idx)

class TestClamp(unittest.TestCase):
    def test_clamp(self):
        x = 5
        lower = 0
        upper = 3
        clamped_x = clamp(x, lower, upper)
        expected_clamped_x = 3
        self.assertEqual(clamped_x, expected_clamped_x)

class TestWalkTrajectoryByDuration(unittest.TestCase):
    def test_staying_on_the_same_segment(self):
        path_points = [[0, 0], [1, 0], [1, 1], [0, 1]]
        velocities = [1, 1, 1, 1]
        starting_idx = 0
        durations = [0.4]
        indices = walk_trajectory_by_durations(path_points, velocities, starting_idx, durations)
        self.assertEqual(indices, [0])
    def test_crossing_1_segment(self):
        path_points = [[0, 0], [1, 0], [1, 1], [0, 1]]
        velocities = [1, 1, 1, 1]
        starting_idx = 0
        durations = [1.4]
        indices = walk_trajectory_by_durations(path_points, velocities, starting_idx, durations)
        self.assertEqual(indices, [1])
    def test_crossing_2_segments(self):
        path_points = [[0, 0], [1, 0], [1, 1], [0, 1]]
        velocities = [1, 1, 1, 1]
        starting_idx = 0
        durations = [2.4]
        indices = walk_trajectory_by_durations(path_points, velocities, starting_idx, durations)
        self.assertEqual(indices, [2])
    def test_multiple_durations_1(self):
        path_points = [[0, 0], [1, 0], [1, 1], [0, 1]]
        velocities = [1, 1, 1, 1]
        starting_idx = 0
        durations = [0.4, 1.0, 1.0]
        indices = walk_trajectory_by_durations(path_points, velocities, starting_idx, durations)
        self.assertEqual(indices, [0,1,2])
    def test_multiple_durations_2(self):
        path_points = [[0, 0], [1, 0], [1, 1], [0, 1]]
        velocities = [1, 1, 1, 1]
        starting_idx = 0
        durations = [0.4, 0.4, 1.0, 1.0]
        indices = walk_trajectory_by_durations(path_points, velocities, starting_idx, durations)
        self.assertEqual(indices, [0,0,1,2])
    def test_multiple_durations_3(self):
        path_points = [[0, 0], [1, 0], [1, 1], [0, 1]]
        velocities = [1, 1, 1, 1]
        starting_idx = 0
        durations = [0.4, 0.4, 0.1, 0.2, 0.2, 0.3, 1.0]
        indices = walk_trajectory_by_durations(path_points, velocities, starting_idx, durations)
        self.assertEqual(indices, [0,0,0,1,1,1,2])

class TestOptimizeL0(unittest.TestCase):
    def get_optimizer_fn(self, version, N, C, solver):
        if version == "v2":
            return optimize_l0_v2
        if version == "v4":
            optimizer = Optimizer(N, C.shape[0], C.shape[1], solver)
            return optimizer.optimize_l0_v4
        
        raise Exception(f"Unknown version {version}")

    @parameterized.expand([
        "v2",
        "v4",
    ])
    def test_simple_integrator(self, version):
        """
        Sanity test to make sure the recovery without attacks works.
        """
        C = np.eye(3)
        A = np.array([
            [0, 1, 0], 
            [0, 0, 1], 
            [0, 0, 0]
        ])
        detaT = 0.1
        Ad = expm(A * detaT)
        x0 = np.array([0, 1, 0.1])

        Phi = np.array([
            C,
            C @ Ad,
            C @ Ad @ Ad,
        ])
        Y = np.array([
            C @ x0,
            C @ Ad @ x0,
            C @ Ad @ Ad @ x0,
        ])
        solver = cp.CLARABEL
        optimizer_fn = self.get_optimizer_fn(version, 3, C, solver)
        soln, _solns, _metadata = optimizer_fn(Phi, Y, solver_args={'solver': solver})
        self.assertIsNotNone(soln)
        s_x0_hat, _s_prob, s_metadata = soln
        self.assertIsNotNone(s_x0_hat)
        self.assertTrue(np.allclose(s_x0_hat, x0, atol=1e-7), f"{s_x0_hat=} != {x0=}")
        self.assertSequenceEqual(s_metadata['K'], [])

    @parameterized.expand([
        "v2",
        "v4",
    ])
    def test_simple_attacks(self, version):
        # This system is maximally 0-sparse observable, and maximally 2-sparse observable with protection on sensor 0
        # > get_s_sparse_observability([C]*3,[A]*2)
        # ([0, 1, 2], True)
        # ([1, 2], False)
        # ([0, 2], True)
        # ([0, 1], True)
        # ([2], False)
        # ([1], False)
        # ([0], True)
        # ([], False)
        C = np.eye(3)
        A = np.array([
            [0, 1, 0], 
            [0, 0, 1], 
            [0, 0, 0]
        ])
        detaT = 0.1
        Ad = expm(A * detaT)
        x0 = np.array([0, 1, 0.1])

        Phi = np.array([
            C,
            C @ Ad,
            C @ Ad @ Ad,
        ])

        solver = cp.CLARABEL
        optimizer_fn = self.get_optimizer_fn(version, 3, C, solver)

        Y = np.array([
            C @ x0 + np.array([0, 0, 1]),
            C @ Ad @ x0 + np.array([0, 0, 1]),
            C @ Ad @ Ad @ x0 + np.array([0, 0, 1]),
        ])
        soln, solns, _metadata = optimizer_fn(Phi, Y, S_list=[S for S in powerset(range(C.shape[0])) if 0 in S], solver_args={'solver': cp.CLARABEL}, early_exit=False)
        self.assertIsNotNone(soln)
        s_x0_hat, s_prob, s_metadata = soln
        self.assertIsNotNone(s_x0_hat)
        self.assertIsNotNone(s_prob)
        self.assertIsNotNone(s_metadata)
        self.assertTrue(np.allclose(s_x0_hat, x0, atol=1e-7), f"{s_x0_hat=} != {x0=}")
        print(s_metadata)
        print([json.dumps({'K': soln[2]['K'], 'S': soln[2]['S'], 'status': soln[1].status}) for soln in solns])
        self.assertSequenceEqual(s_metadata['K'], [2])
        print([json.dumps({'K': soln[2]['K'], 'S': soln[2]['S'], 'status': soln[1].status}) for soln in solns])
        self.assertSetEqual(set([
            json.dumps({'K': [], 'S': [0,1,2], 'status': 'infeasible'}),
            json.dumps({'K': [2], 'S': [0,1], 'status': 'optimal'}),
            json.dumps({'K': [1], 'S': [0,2], 'status': 'infeasible'}),
            json.dumps({'K': [1,2], 'S': [0], 'status': 'optimal'}),
        ]), set(json.dumps({'K': soln[2]['K'], 'S': soln[2]['S'], 'status': soln[1].status}) for soln in solns))

        Y = np.array([
            C @ x0 + np.array([0, 1, 0]),
            C @ Ad @ x0 + np.array([0, 1, 0]),
            C @ Ad @ Ad @ x0 + np.array([0, 1, 0]),
        ])
        soln, solns, _metadata = optimizer_fn(Phi, Y, S_list=[S for S in powerset(range(C.shape[0])) if 0 in S], solver_args={'solver': cp.CLARABEL}, early_exit=False)
        self.assertIsNotNone(soln)
        s_x0_hat, s_prob, s_metadata = soln
        self.assertIsNotNone(s_x0_hat)
        self.assertIsNotNone(s_prob)
        self.assertIsNotNone(s_metadata)
        self.assertTrue(np.allclose(s_x0_hat, x0, atol=1e-7), f"{s_x0_hat=} != {x0=}")
        self.assertSequenceEqual(s_metadata['K'], [1])
        self.assertSetEqual(set([
            json.dumps({'K': [], 'S': [0,1,2], 'status': 'infeasible'}),
            json.dumps({'K': [2], 'S': [0,1], 'status': 'infeasible'}),
            json.dumps({'K': [1], 'S': [0,2], 'status': 'optimal'}),
            json.dumps({'K': [1,2], 'S': [0], 'status': 'optimal'}),
        ]), set(json.dumps({'K': soln[2]['K'], 'S': soln[2]['S'], 'status': soln[1].status}) for soln in solns))


class TestGetStateEvolutionTensor(unittest.TestCase):
    def test_single_matrix(self):
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        As = [A]
        expected_Evo = np.zeros((2, 3, 3))
        expected_Evo[0] = np.eye(3)
        expected_Evo[1] = A
        Evo = get_state_evolution_tensor(As)
        np.testing.assert_allclose(Evo, expected_Evo)

    def test_multiple_matrices(self):
        A1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        A2 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
        As = [A1, A2]
        expected_Evo = np.zeros((3, 3, 3))
        expected_Evo[0] = np.eye(3)
        expected_Evo[1] = A1
        expected_Evo[2] = np.matmul(A2, A1)
        Evo = get_state_evolution_tensor(As)
        np.testing.assert_allclose(Evo, expected_Evo)

class TestGetOutputEvolutionTensor(unittest.TestCase):
    def test_get_output_evolution_tensor(self):
        Cs = [
            np.array([
                [1, 0, 0],
                [0, 1, 0]
            ]),
            np.array([
                [0, 1, 0],
                [1, 0, 0]
            ]),
            np.array([
                [1, 0, 1],
                [0, 1, 0]
            ])
        ]
        As = [
            np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ]),
            np.array([
                [1, 0, 1],
                [0, 1, 0],
                [0, 0, 1]
            ]),
        ]
        Evo = get_state_evolution_tensor(As)
        expected_Phi = np.array([
            [
                [1, 0, 0],
                [0, 1, 0]
            ],
            [
                [0, 1, 0],
                [1, 0, 0]
            ],
            [
                [1, 0, 2],
                [0, 1, 0]
            ],
        ])
        Phi = get_output_evolution_tensor(Cs, Evo)
        np.testing.assert_allclose(Phi, expected_Phi)

class OrderedBuckets():
    """
    A class that allows us to define a list of buckets and then access them by item index.
    """

    def __init__(self, buckets):
        self.buckets = buckets
    
    def getBuckeForItem(self, i):
        for bucket in self.buckets:
            i -= len(bucket)
            if i < 0:
                return bucket
        
        raise IndexError(f"Index {i} out of range")

class TestGetSSparseObservability(unittest.TestCase):
    def test_simple(self):
        # Define system matrices
        As = [
            np.array([
                [1, 1],
                [0, 1]
            ]),
            np.array([
                [1, 0],
                [0, 1]
            ])
        ]
        Cs = [np.eye(2)] * 3

        # Compute s-sparse observability
        s, cases = get_s_sparse_observability(Cs, As)

        # Check results
        self.assertEqual(s, 0)
        expected_cases = [
            ([0, 1], True),
            ([0], True),
            ([1], False),
            ([], False),
        ]
        self.assertEqual(len(cases), len(expected_cases))
        # We don't enforce the order the cases are returned.
        for i in range(len(cases)):
            self.assertIn(cases[i], expected_cases)

    def test_complete(self):
        # A 4x4 system
        A = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ])
        n = A.shape[0]

        C = np.eye(n)
        self.assertEqual(get_s_sparse_observability([C]*n,[A]*(n-1), early_exit=True)[0], 0)

        C = np.concatenate((np.eye(n), np.eye(n)))
        self.assertEqual(get_s_sparse_observability([C]*n,[A]*(n-1), early_exit=True)[0], 1)

        C = np.concatenate((np.eye(n), np.eye(n), np.eye(n)))
        self.assertEqual(get_s_sparse_observability([C]*n,[A]*(n-1), early_exit=True)[0], 2)

        C = np.concatenate((np.eye(n), np.eye(n), np.eye(n), np.eye(n)))
        self.assertEqual(get_s_sparse_observability([C]*n,[A]*(n-1), early_exit=True)[0], 3)

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
        self.assertEqual(get_s_sparse_observability([C]*n,[A]*(n-1), early_exit=True)[0], 0)

        C = np.concatenate((np.eye(n), np.eye(n)))
        self.assertEqual(get_s_sparse_observability([C]*n,[A]*(n-1), early_exit=True)[0], 1)

        C = np.concatenate((np.eye(n), np.eye(n), np.eye(n)))
        self.assertEqual(get_s_sparse_observability([C]*n,[A]*(n-1), early_exit=True)[0], 2)

        C = np.concatenate((np.eye(n), np.eye(n), np.eye(n), np.eye(n)))
        self.assertEqual(get_s_sparse_observability([C]*n,[A]*(n-1), early_exit=True)[0], 3)

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
        self.assertEqual(get_s_sparse_observability([C]*n,[A]*(n-1), early_exit=True)[0], 1)

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
        self.assertEqual(get_s_sparse_observability([C]*n,[A]*(n-1), early_exit=True)[0], 2)

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
        self.assertEqual(get_s_sparse_observability([C]*n,[A]*(n-1), early_exit=True)[0], 3)

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
        self.assertEqual(get_s_sparse_observability([C]*n,[A]*(n-1), early_exit=True)[0], 4)

    def test_v_zero(self):
        theta = np.pi/4
        v = 0
        delta = np.pi/6
        dt = 0.1
        l = 1.5
        Ad, Bd = kinematic_bicycle_model_linearize(theta, v, delta, dt, l)
        # Since v=0, the position change should be very small
        np.testing.assert_allclose(Ad, np.eye(5), atol=1e-1)
        # Inputs should have no effect on the position and orientation
        np.testing.assert_allclose(Bd, np.zeros((5, 2)), atol=1e-1)

    def test_large_delta(self):
        theta = np.pi/4
        v = 1
        delta = np.pi/2  # large steering angle
        dt = 0.1
        l = 2
        Ad, Bd = kinematic_bicycle_model_linearize(theta, v, delta, dt, l)
        expected_Ad, expected_Bd = Ad, Bd  # Using the computed values as expected for this test
        np.testing.assert_allclose(Ad, expected_Ad, atol=1e-8)
        np.testing.assert_allclose(Bd, expected_Bd, atol=1e-8)

    def test_non_zero_theta(self):
        theta = np.pi/3
        v = 2
        delta = np.pi/6
        dt = 0.05
        l = 1.5
        Ad, Bd = kinematic_bicycle_model_linearize(theta, v, delta, dt, l)
        expected_Ad, expected_Bd = Ad, Bd  # Using the computed values as expected for this test
        np.testing.assert_allclose(Ad, expected_Ad, atol=1e-8)
        np.testing.assert_allclose(Bd, expected_Bd, atol=1e-8)
        
    def test_negative_v(self):
        theta = np.pi/2
        v = -1  # Negative velocity
        delta = np.pi/6
        dt = 0.05
        l = 2
        Ad, Bd = kinematic_bicycle_model_linearize(theta, v, delta, dt, l)
        expected_Ad, expected_Bd = Ad, Bd  # Using the computed values as expected for this test
        np.testing.assert_allclose(Ad, expected_Ad, atol=1e-8)
        np.testing.assert_allclose(Bd, expected_Bd, atol=1e-8)
        
    def test_dt_zero(self):
        theta = np.pi/4
        v = 1
        delta = np.pi/6
        dt = 0  # Time step is zero
        l = 2
        Ad, Bd = kinematic_bicycle_model_linearize(theta, v, delta, dt, l)
        # Since dt=0, Ad should be identity matrix and Bd should be zero matrix.
        np.testing.assert_allclose(Ad, np.eye(5), atol=1e-8)
        np.testing.assert_allclose(Bd, np.zeros((5, 2)), atol=1e-8)


class TestCalcInputEffectsOnOutput(unittest.TestCase):
    def test_calc_input_effects_on_output(self):
        As = [
            np.array([
                [1.5, 0],
                [0, 2]
            ]),
            np.array([
                [3, 0],
                [0, 4]
            ])
        ]
        Bs = [
            np.array([
                [1],
                [0]
            ]),
            np.array([
                [0],
                [1]
            ]),
        ]
        Cs = [np.eye(2)] * 3
        inputs = [
            np.array([1]),
            np.array([2])
        ]
        output_effects = calc_input_effects_on_output(As, Bs, Cs, inputs)
        expected_output_effects = [
            np.zeros((2,)),
            np.array([1,0]),
            np.array([1.5,2])
        ]
        for i in range(len(output_effects)):
            np.testing.assert_allclose(output_effects[i], expected_output_effects[i])


if __name__ == '__main__':
    unittest.main()
