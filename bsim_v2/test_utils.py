import unittest
import numpy as np
from math import pi, sin, cos, atan2, sqrt
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

if __name__ == '__main__':
    unittest.main()