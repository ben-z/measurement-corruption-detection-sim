from math import atan2

import numpy as np
from scipy.linalg import expm

from ..utils import wrap_to_pi
from .base_model import BaseModel


class KinematicBicycle5StateRearWheelRefModel(BaseModel):
    """
    5-state rear-wheel reference kinematic bicycle model.
    States:
        x: x position in m
        y: y position in m
        theta: heading in rad
        v: speed in m/s
        delta: steering angle in rad
    Inputs:
        a: acceleration in m/s^2
        delta_dot: steering rate in rad/s
    """
    num_states = 5
    num_inputs = 2

    def __init__(self, L=2.9, max_steer=0.5, max_speed=1.0, max_accel=1.0, max_steer_rate=1.0):
        """
        Creates a 5-state rear-wheel reference kinematic bicycle model.
        :param x0: initial state [x, y, theta, v, delta]
        :param dt: time step in seconds
        :param L: wheelbase in m
        :param max_steer: maximum steering angle in radians
        :param max_speed: maximum speed in m/s
        """
        self.L = L
        self.max_steer = max_steer
        self.max_speed = max_speed

    def next(self, state, dt, u):
        """
        Updates the state of the model.
        """
        x, y, theta, v, delta = state
        a, delta_dot = u

        v = np.clip(v + a * dt, 0, self.max_speed)
        delta = np.clip(delta + delta_dot * dt, -self.max_steer, self.max_steer)
        theta = theta + v / self.L * np.tan(delta) * dt
        x = x + v * np.cos(theta) * dt
        y = y + v * np.sin(theta) * dt

        return np.array([x, y, theta, v, delta])

    def linearize(self, state, u, dt):
        """
        Linearizes the model around the current state and input.
        :return: Discrete-time state-space matrices Ad and Bd
        """
        x, y, theta, v, delta = state
        a, delta_dot = u
        dt = dt
        L = self.L

        A = np.array(
            [
                [0, 0, -v * np.sin(theta), np.cos(theta), 0],
                [0, 0, v * np.cos(theta), np.sin(theta), 0],
                [0, 0, 0, np.tan(delta) / L, v * (1 + np.tan(delta) ** 2) / L],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )

        B = np.array(
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [1, 0],
                [0, 1],
            ]
        )

        # Calculate Ad and Bd at the same time: https://en.wikipedia.org/wiki/Discretization#cite_note-2
        ABd = expm(
            dt * np.block([[A, B], [np.zeros((B.shape[1], A.shape[0] + B.shape[1]))]])
        )

        Ad = ABd[: A.shape[0], : A.shape[1]]
        Bd = ABd[: B.shape[0], A.shape[1] :]

        return Ad, Bd

    def subtract_states(self, x1, x2):
        """
        Subtracts two states. Handles the wrap-around of angular quantities.
        """
        x1_x, x1_y, x1_theta, x1_v, x1_delta = x1
        x2_x, x2_y, x2_theta, x2_v, x2_delta = x2

        return np.array([
            x1_x - x2_x,
            x1_y - x2_y,
            wrap_to_pi(x1_theta - x2_theta),
            x1_v - x2_v,
            wrap_to_pi(x1_delta - x2_delta),
        ])

    def state_mean(self, sigmas, Wm):
        """
        Compute the mean of a set of sigma points.
        Parameters:
            sigmas: ndarray, of size (n, 2n+1)
                2D array of sigma points.
            Wm: ndarray, of size (n,)
                Weights for the mean.
        """
        x = np.zeros(5)
        sum_sin = np.sum(np.dot(np.sin(sigmas[:, 2]), Wm))
        sum_cos = np.sum(np.dot(np.cos(sigmas[:, 2]), Wm))
        x[0] = np.dot(sigmas[:, 0], Wm)
        x[1] = np.dot(sigmas[:, 1], Wm)
        x[2] = atan2(sum_sin, sum_cos)
        x[3] = np.dot(sigmas[:, 3], Wm)
        x[4] = np.dot(sigmas[:, 4], Wm)
        return x
