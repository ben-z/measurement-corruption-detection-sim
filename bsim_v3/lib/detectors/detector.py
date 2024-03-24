import time

import cvxpy as cp
import numpy as np

from ..planners.utils import closest_point_idx, closest_point_idx_local
from .optimizer import Optimizer
from .utils import (calc_input_effects_on_output, get_output_evolution_tensor,
                    get_state_evolution_tensor, walk_trajectory_by_durations)


class Detector:
    def __init__(self, model, sensor, N, dt, eps):
        assert len(eps) == sensor.num_outputs, f"{len(eps)} != {sensor.num_outputs=}"

        self.model = model
        self.sensor = sensor
        self.N = N
        self.dt = dt
        self.eps = eps
        self.optimizer = Optimizer(N, sensor.num_outputs, model.num_states)
        self.Cs = []
        self.us = []
        self.ys = []
        self.estimates = []
        self.plans = []
        self.validity = np.ones(self.sensor.num_outputs, dtype=bool)

    def step(self, y_k, estimate_k, plan_k, u_k):
        """
        Run the detector step. Records data for the optimizer.
        Parameters:
            y_k: The sensor output at time k (current time)
            estimate_k: The state estimate at time k (current time)
            plan_k: The plan at time k (current time)
            u_k: The input at time k (current time)
        """
        self.Cs.append(self.sensor.get_output_matrix())
        self.ys.append(y_k)
        self.us.append(u_k)
        self.estimates.append(estimate_k)
        self.plans.append(plan_k)

        if len(self.Cs) > self.N:
            self.Cs.pop(0)
            self.us.pop(0)
            self.ys.pop(0)
            self.estimates.pop(0)
            self.plans.pop(0)

    def get_linearization_points(self):
        """
        Generate linearization points for the optimizer.
        """
        prev_plan = None
        prev_closest_idx = None

        for estimate, plan in zip(self.estimates, self.plans):
            if prev_plan is not None and plan.id == prev_plan.id:
                # The plan hasn't changed, so we can localize using
                # the previous closest point index.
                idx = closest_point_idx_local(plan.points, estimate[0], estimate[1], prev_closest_idx)
            else:
                idx = closest_point_idx(plan.points, estimate[0], estimate[1])

            prev_plan = plan
            prev_closest_idx = idx

            linearization_point = np.array([
                plan.points[idx][0],
                plan.points[idx][1],
                plan.headings[idx],
                plan.velocities[idx],
                0.0
            ])

            # To stay on the path, we apply zero acceleration and zero steering
            yield linearization_point, np.array([0,0])

    def get_optimizer_params(self):
        """
        Get the parameters for the optimizer.
        Override this method to use custom methods for generating the optimizer parameters.
        """

        linearization_points = list(self.get_linearization_points())
        desired_output_trajectory = [self.sensor.get_output(state) for state, _ in linearization_points]
        models = [self.model.linearize(state, u, self.dt) for state, u in linearization_points[:-1]]

        As = list(m[0] for m in models)
        Bs = list(m[1] for m in models)

        input_effects = calc_input_effects_on_output(As, Bs, self.Cs, self.us[:-1])

        output_hist_no_input_effects = [
            self.sensor.normalize_output(output - input_effect - desired_output)
            for output, input_effect, desired_output in zip(
                self.ys, input_effects, desired_output_trajectory
            )
        ]

        Y = np.array(output_hist_no_input_effects)
        Phi = get_output_evolution_tensor(self.Cs, get_state_evolution_tensor(As))

        return Phi, Y

    def calc_validity(self):
        """
        Run the detector and return the validity of the sensor outputs.
        """

        start = time.perf_counter()
        if len(self.Cs) < self.N:
            # Not enough data to run the detector
            return self.validity

        Phi, Y = self.get_optimizer_params()

        optimizer_res = self.optimizer.optimize_l0_v4(
            Phi, Y, self.eps, solver_args={"solver": cp.CLARABEL}
        )

        end = time.perf_counter()

        if optimizer_res.soln is not None:
            prob = optimizer_res.soln.prob
            res_metadata = optimizer_res.soln.metadata
            optimizer_metadata = optimizer_res.metadata
            print(f"{prob.status}, K={res_metadata['K']}, total took {end-start:.4f} s, optimizer took {optimizer_metadata['solve_time']:.4f} s")

            validity = np.ones(self.sensor.num_outputs, dtype=bool)
            validity[res_metadata["K"]] = False
            self.validity = validity



        return self.validity

class LookAheadDetector(Detector):
    """
    A detector that uses a plan look-ahead approach (based on the estimate atthe beginning of the window)
    to generate linearization points.
    """
    def get_linearization_points(self):
        """
        Generate linearization points for the optimizer.
        """
        initial_estimate = self.estimates[0]
        initial_closest_idx = closest_point_idx(self.plans[0].points, initial_estimate[0], initial_estimate[1])

        desired_path_indices = [initial_closest_idx] + walk_trajectory_by_durations(
            self.plans[0].points, self.plans[0].velocities, initial_closest_idx, [self.dt] * (self.N - 1)
        )

        for idx in desired_path_indices:
            plan = self.plans[0]
            linearization_point = np.array([
                plan.points[idx][0],
                plan.points[idx][1],
                plan.headings[idx],
                plan.velocities[idx],
                0.0
            ])

            # To stay on the path, we apply zero acceleration and zero steering
            yield linearization_point, np.array([0,0])
