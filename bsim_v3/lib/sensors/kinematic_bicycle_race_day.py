from math import atan2

import numpy as np

from .base_sensor import BaseSensor
from ..utils import wrap_to_pi


class KinematicBicycleRaceDaySensor(BaseSensor):
    _num_states = 5 # x, y, theta, v, delta
    _num_outputs = 6 # x, y, theta, v1, v2, delta
    _C = np.array([
        [1, 0, 0, 0, 0], # x
        [0, 1, 0, 0, 0], # y
        [0, 0, 1, 0, 0], # theta
        [0, 0, 0, 1, 0], # v1
        [0, 0, 0, 1, 0], # v2
        [0, 0, 0, 0, 1], # delta
    ])

    def get_output(self, state):
        return self._C @ state

    def output_mean(self, sigmas, Wm):
        z = np.zeros(6)

        sum_sin = np.sum(np.dot(np.sin(sigmas[:, 2]), Wm))
        sum_cos = np.sum(np.dot(np.cos(sigmas[:, 2]), Wm))
        z[0] = np.dot(sigmas[:, 0], Wm)
        z[1] = np.dot(sigmas[:, 1], Wm)
        z[2] = atan2(sum_sin, sum_cos)
        z[3] = np.dot(sigmas[:, 3], Wm)
        z[4] = np.dot(sigmas[:, 4], Wm)
        z[5] = np.dot(sigmas[:, 5], Wm)
        return z

    def subtract_outputs(self, o1, o2):
        o1_x, o1_y, o1_theta, o1_v1, o1_v2, o1_delta = o1
        o2_x, o2_y, o2_theta, o2_v1, o2_v2, o2_delta = o2

        return np.array([
            o1_x - o2_x,
            o1_y - o2_y,
            wrap_to_pi(o1_theta - o2_theta),
            o1_v1 - o2_v1,
            o1_v2 - o2_v2,
            wrap_to_pi(o1_delta - o2_delta),
        ])
    
    def get_output_matrix(self):
        return self._C
    
    def normalize_output(self, o):
        o[2] = wrap_to_pi(o[2])
        o[5] = wrap_to_pi(o[5])
        return o