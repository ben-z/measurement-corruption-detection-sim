import numpy as np
from scipy.linalg import expm

from .base_plant import BasePlant

class KinematicBicycle5StateRearWheelRef(BasePlant):
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
    def __init__(self, x0, dt, L=2.9, max_steer=0.5, max_speed=1.0):
        """
        Creates a 5-state rear-wheel reference kinematic bicycle model.
        :param x0: initial state [x, y, theta, v, delta]
        :param dt: time step in seconds
        :param L: wheelbase in m
        :param max_steer: maximum steering angle in radians
        :param max_speed: maximum speed in m/s
        """
        x0 = np.array(x0)
        assert x0.shape == (5,)

        super().__init__(x0, dt)
        self.L = L
        self.max_steer = max_steer
        self.max_speed = max_speed
        self.u = np.zeros(2)

    def set_inputs(self, a, delta_dot):
        """
        Sets the inputs to the model.
        :param a: acceleration in m/s^2
        :param delta_dot: steering rate in rad/s
        """
        self.u = np.array([a, delta_dot])

    def next(self):
        """
        Updates the state of the model.
        """
        x, y, theta, v, delta = self.x
        a, delta_dot = self.u

        v = np.clip(v + a * self.dt, 0, self.max_speed)
        delta = np.clip(delta + delta_dot * self.dt, -self.max_steer, self.max_steer)
        theta = theta + v / self.L * np.tan(delta) * self.dt
        x = x + v * np.cos(theta) * self.dt
        y = y + v * np.sin(theta) * self.dt

        self.x = np.array([x, y, theta, v, delta])

    def linearize(self):
        """
        Linearizes the model around the current state and input.
        :return: Discrete-time state-space matrices Ad and Bd
        """
        x, y, theta, v, delta = self.x
        a, delta_dot = self.u
        dt = self.dt
        L = self.L

        A = np.array([
            [0, 0, -v*np.sin(theta), np.cos(theta), 0],
            [0, 0, v*np.cos(theta), np.sin(theta), 0],
            [0, 0, 0, np.tan(delta)/L, v*(1+np.tan(delta)**2)/L],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ])

        B = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 1],
        ])

        # Calculate Ad and Bd at the same time: https://en.wikipedia.org/wiki/Discretization#cite_note-2
        ABd = expm(dt * np.block([
            [A, B],
            [np.zeros((B.shape[1], A.shape[0]+B.shape[1]))]
        ]))

        Ad = ABd[:A.shape[0], :A.shape[1]]
        Bd = ABd[:B.shape[0], A.shape[1]:]

        return Ad, Bd


