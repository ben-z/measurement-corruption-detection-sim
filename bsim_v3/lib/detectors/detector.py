import time

import cvxpy as cp
import numpy as np
from typing import NamedTuple, Optional

from ..planners.utils import closest_point_idx, closest_point_idx_local
from .optimizer import Optimizer, OptimizerMetadata
from .utils import (calc_input_effects_on_output, get_output_evolution_tensor,
                    get_state_evolution_tensor, walk_trajectory_by_durations)

class CalcValidityMetadata(NamedTuple):
    total_time: float
    prob_status: Optional[str]
    optimizer_metadata: Optional[OptimizerMetadata]


class Detector:
    def __init__(self, model, sensor, N, dt, eps, S_list):
        """
        Initialize the detector.
        Parameters:
            model: The plant model
            sensor: The sensor
            N: The window size
            dt: The time step
            eps: The noise tolerance
            S_list: The list of sensor combinations to try to check if the combination is valid. If a sensor is protected, all sets containing the sensor should exist in S_list.
        """
        assert len(eps) == sensor.num_outputs, f"{len(eps)} != {sensor.num_outputs=}"

        self.model = model
        self.sensor = sensor
        self.N = N
        self.dt = dt
        self.eps = eps
        self.S_list = S_list
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

            linearization_state = np.array([
                plan.points[idx][0],
                plan.points[idx][1],
                plan.headings[idx],
                plan.velocities[idx],
                0.0
            ])

            # To stay on the path, we apply zero acceleration and zero steering
            yield linearization_state, np.array([0,0])

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

    def calc_validity(self) -> tuple[np.ndarray, CalcValidityMetadata]:
        """
        Run the detector and return the validity of the sensor outputs.
        """

        if len(self.Cs) < self.N:
            # Not enough data to run the detector
            return self.validity, CalcValidityMetadata(total_time=0, prob_status=None, optimizer_metadata=None)

        start = time.perf_counter()

        Phi, Y = self.get_optimizer_params()

        optimizer_res = self.optimizer.optimize_l0_v4(
            Phi, Y, self.eps, solver_args={"solver": cp.CLARABEL}, S_list=self.S_list,
        )

        end = time.perf_counter()

        if optimizer_res.soln is not None:
            prob = optimizer_res.soln.prob
            prob_status = prob.status
            res_metadata = optimizer_res.soln.metadata
            optimizer_metadata = optimizer_res.metadata
            # print(f"{prob_status}, K={res_metadata['K']}, total took {end-start:.4f} s, optimizer took {optimizer_metadata['solve_time']:.4f} s")

            validity = np.ones(self.sensor.num_outputs, dtype=bool)
            validity[res_metadata.K] = False
            self.validity = validity
        else:
            prob_status = None
            optimizer_metadata = None

        return self.validity, CalcValidityMetadata(
            total_time=end - start,
            prob_status=prob_status,
            optimizer_metadata=optimizer_metadata,
        )

class EveryEstimateDetector(Detector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._desired_outputs = []
        self._As = []
        self._Bs = []
        self._prev_closest_idx = None
        self._prev_plan = None

    def step(self, y_k, estimate_k, plan_k, u_k):
        super().step(y_k, estimate_k, plan_k, u_k)

        prev_closest_idx = self._prev_closest_idx
        prev_plan = self._prev_plan
        plan = plan_k
        estimate = estimate_k

        if prev_plan is not None and plan.id == prev_plan.id:
            # The plan hasn't changed, so we can localize using
            # the previous closest point index.
            idx = closest_point_idx_local(plan.points, estimate[0], estimate[1], prev_closest_idx)
        else:
            idx = closest_point_idx(plan.points, estimate[0], estimate[1])

        self._prev_plan = plan
        self._prev_closest_idx = idx

        linearization_state = np.array([
            plan.points[idx][0],
            plan.points[idx][1],
            plan.headings[idx],
            plan.velocities[idx],
            0.0
        ])
        # assuming straight line segments
        linearization_input = np.array([0,0])

        desired_output = self.sensor.get_output(linearization_state)
        A, B = self.model.linearize(linearization_state, linearization_input, self.dt)

        self._desired_outputs.append(desired_output)
        self._As.append(A)
        self._Bs.append(B)

        if len(self._desired_outputs) > self.N:
            self._desired_outputs.pop(0)
            self._As.pop(0)
            self._Bs.pop(0)

    def get_linearization_points(self):
        raise NotImplementedError("This class does not use the get_linearization_points method.")

    def get_optimizer_params(self):
        """
        Get the parameters for the optimizer.
        Override this method to use custom methods for generating the optimizer parameters.
        """

        As = self._As[:-1]
        Bs = self._Bs[:-1]
        desired_output_trajectory = self._desired_outputs

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

class LookAheadDetector(Detector):
    """
    A detector that uses a plan look-ahead approach (based on the estimate atthe beginning of the window)
    to generate linearization points.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prev_closest_idx = None

    def get_linearization_points(self):
        """
        Generate linearization points for the optimizer.
        """
        initial_estimate = self.estimates[0]
        if self._prev_closest_idx is None:
            initial_closest_idx = closest_point_idx(self.plans[0].points, initial_estimate[0], initial_estimate[1])
        else:
            initial_closest_idx = closest_point_idx_local(self.plans[0].points, initial_estimate[0], initial_estimate[1], self._prev_closest_idx)
        
        self._prev_closest_idx = initial_closest_idx

        desired_path_indices = [initial_closest_idx] + walk_trajectory_by_durations(
            self.plans[0].points, self.plans[0].velocities, initial_closest_idx, [self.dt] * (self.N - 1)
        )

        for idx in desired_path_indices:
            plan = self.plans[0]
            linearization_state = np.array([
                plan.points[idx][0],
                plan.points[idx][1],
                plan.headings[idx],
                plan.velocities[idx],
                0.0
            ])

            # To stay on the path, we apply zero acceleration and zero steering
            yield linearization_state, np.array([0,0])
