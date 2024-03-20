import numpy as np
from scipy.linalg import expm

from .base_plant import BasePlant
from ..models.kinematic_bicycle import KinematicBicycle5StateRearWheelRefModel

class KinematicBicycle5StateRearWheelRefPlant(BasePlant):
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
    def __init__(self, x0, dt, L=2.9, max_steer=0.5, max_speed=1.0, max_accel=1.0, max_steer_rate=1.0):
        """
        Creates a 5-state rear-wheel reference kinematic bicycle model.
        :param x0: initial state [x, y, theta, v, delta]
        :param dt: time step in seconds
        :param L: wheelbase in m
        :param max_steer: maximum steering angle in radians
        :param max_speed: maximum speed in m/s
        """
        self.model = KinematicBicycle5StateRearWheelRefModel(
            L=L,
            max_steer=max_steer,
            max_speed=max_speed,
            max_accel=max_accel,
            max_steer_rate=max_steer_rate,
        )

        x0 = np.array(x0)
        assert x0.shape == (self.model.num_states,), f"{x0.shape=} != {(self.model.num_states,)=}"

        super().__init__(x0, dt)
        self.L = L
        self.max_steer = max_steer
        self.max_speed = max_speed
        self.u = np.zeros(self.model.num_inputs)

    def set_inputs(self, u):
        """
        Sets the inputs to the model.
        :param a: acceleration in m/s^2
        :param delta_dot: steering rate in rad/s
        """
        assert len(u) == self.model.num_inputs, f"{len(u)=} != {(self.model.num_inputs,)=}"

        self.u = np.array(u)

    def next(self):
        """
        Updates the state of the model.
        """
        self.x = self.model.next(self.x, self.dt, self.u)
