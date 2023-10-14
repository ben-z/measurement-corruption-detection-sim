import unittest
import numpy as np
import cvxpy as cp
from math import pi, sin, cos, atan2, sqrt, tan
from scipy.linalg import expm
from utils import (
    kinematic_bicycle_model,
    generate_circle_approximation,
    generate_figure_eight_approximation,
    PIDController,
    closest_point_idx,
    closest_point_idx_local,
    wrap_to_pi,
    get_lookahead_idx,
    clamp,
    walk_trajectory_by_durations,
    optimize_l0,
    get_state_evolution_tensor,
    get_output_evolution_tensor,
    get_s_sparse_observability,
)

class TestKinematicBicycleModel(unittest.TestCase):
    def test_noop(self):
        state = np.array([0, 0, 0, 0, 0])
        input = np.array([0, 0])
        params = {'dt': 0.1, 'l': 1}
        new_state = kinematic_bicycle_model(state, input, params)
        expected_state = np.array([0, 0, 0, 0, 0])
        np.testing.assert_allclose(new_state, expected_state)

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
    def test_simple_integrator(self):
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
        x0_hat, prob, metadata, solns = optimize_l0(Phi, Y)

        self.assertIsNotNone(x0_hat)
        self.assertIsNotNone(prob)
        self.assertIsNotNone(metadata)
        self.assertTrue(np.allclose(x0_hat.value, x0), f"{x0_hat.value=} != {x0=}")
        self.assertSequenceEqual(metadata['K'], [])

    def test_simple_attacks(self):
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
            C @ x0 + np.array([0, 0, 1]),
            C @ Ad @ x0 + np.array([0, 0, 1]),
            C @ Ad @ Ad @ x0 + np.array([0, 0, 1]),
        ])
        x0_hat, prob, metadata, solns = optimize_l0(Phi, Y)
        self.assertIsNotNone(x0_hat)
        self.assertIsNotNone(prob)
        self.assertIsNotNone(metadata)
        self.assertTrue(np.allclose(x0_hat.value, x0), f"{x0_hat.value=} != {x0=}")
        self.assertSequenceEqual(metadata['K'], [2])

        # TODO: find the s-sparse observability of the simple integrator and test this on more complex systems
        # In this example, losing sensor 0 decreases observability. So we need to protect 0.
        # > get_s_sparse_observability([C]*3,[A]*2)
        # ([0, 1, 2], True)
        # ([1, 2], False)
        # ([0, 2], True)
        # ([0, 1], True)
        # ([2], False)
        # ([1], False)
        # ([0], True)
        # ([], False)
        # Current feasibility status:
        # > [print(soln[2]['S'], soln[1].status) for soln in solns]
        # () infeasible
        # (0,) optimal
        # (1,) optimal
        # (2,) infeasible
        # (0, 1) optimal
        # (0, 2) optimal
        # (1, 2) optimal
        # (0, 1, 2) optimal
        # Y = np.array([
        #     C @ x0 + np.array([0, 1, 0]),
        #     C @ Ad @ x0 + np.array([0, 1, 0]),
        #     C @ Ad @ Ad @ x0 + np.array([0, 1, 0]),
        # ])
        # x0_hat, prob, metadata, solns = optimize_l0(Phi, Y)
        # self.assertIsNotNone(x0_hat)
        # self.assertIsNotNone(prob)
        # self.assertIsNotNone(metadata)
        # self.assertTrue(np.allclose(x0_hat.value, x0), f"{x0_hat.value=} != {x0=}")
        # self.assertSequenceEqual(metadata['K'], [1])

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

if __name__ == '__main__':
    unittest.main()
