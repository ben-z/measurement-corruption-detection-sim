from math import atan2, sin, sqrt

import numpy as np

from ..planners.base_planner import PlannerOutput
from ..planners.utils import (closest_point_idx, closest_point_idx_local,
                             get_lookahead_idx)
from .base_controller import BaseController
from .pid import PIDController


class KinematicBicycle5StatePurePursuitController(BaseController):
    def __init__(self, L, max_steer_rate, max_accel, lookahead_fn, dt):
        self.L = L
        self.max_steer_rate = max_steer_rate
        self.max_accel = max_accel
        self.lookahead = lookahead_fn
        self._last_plan_id = None
        self._prev_closest_idx = 0

        self.a_controller = PIDController(5, 0.0, 0.0, dt)
        self.delta_dot_controller = PIDController(30, 0.0, 0.0, dt)

    def step(self, plan: PlannerOutput, estimate):
        x, y, theta, v, delta = estimate

        if self._last_plan_id == plan.id:
            # The plan hasn't changed, so we can localize using
            # the previous closest point index.
            idx = closest_point_idx_local(plan.points, x, y, self._prev_closest_idx)
        else:
            # The plan has changed or this is the first step, so we
            # need to search the entire path.
            idx = closest_point_idx(plan.points, x, y)
            self._last_plan_id = plan.id

        assert idx is not None, "No closest point found."
        self._prev_closest_idx = idx
        
        # Find the lookahead point
        lookahead_dist = self.lookahead(v)
        target_idx = get_lookahead_idx(plan.points, idx, lookahead_dist)

        target_point = plan.points[target_idx]
        target_velocity = plan.velocities[target_idx]

        dist_to_target = sqrt((target_point[0] - x)**2 + (target_point[1] - y)**2)
        angle_to_target = atan2(target_point[1] - y, target_point[0] - x) - theta
        target_delta = atan2(2*self.L*sin(angle_to_target), dist_to_target)

        a = np.clip(self.a_controller.step(target_velocity - v), -self.max_accel, self.max_accel)
        delta_dot = np.clip(self.delta_dot_controller.step(target_delta - delta), -self.max_steer_rate, self.max_steer_rate)

        return np.array([a, delta_dot]), {
            "target_point": target_point,
            "target_heading": plan.headings[target_idx],
            "target_velocity": target_velocity,
            "target_idx": target_idx,
            "dist_to_target": dist_to_target,
            "angle_to_target": angle_to_target,
            "target_delta": target_delta,
        }



