from typing import Optional

import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter

from ..models.base_model import BaseModel
from ..sensors.base_sensor import BaseSensor
from .base_estimator import BaseEstimator


class SimpleUKF(BaseEstimator):
    def __init__(
        self,
        model: BaseModel,
        sensor: BaseSensor,
        dt: float,
        x0: np.ndarray,
        P: np.ndarray,
        R: np.ndarray,
        Q: np.ndarray,
        alpha: float = 0.3,
        beta: float = 2,
    ):
        super().__init__(model, dt)
        self.sensor = sensor

        assert x0.shape == (model.num_states,), f"{x0.shape} != {(model.num_states,)=}"

        ukf_sigma_points = MerweScaledSigmaPoints(
            n=model.num_states,
            alpha=alpha,
            beta=beta,
            kappa=3 - model.num_states, # good rule of thumb according to filterpy
            subtract=model.subtract_states,
        )
        ukf = UnscentedKalmanFilter(
            dim_x=model.num_states,
            dim_z=sensor.num_outputs,
            dt=dt,
            fx=model.next,
            hx=sensor.get_output,
            points=ukf_sigma_points,
            residual_x=model.subtract_states,
            residual_z=sensor.subtract_outputs,
            x_mean_fn=model.state_mean,
            z_mean_fn=sensor.output_mean,
        )
        ukf.x = x0
        ukf.P = P # initial state covariance
        ukf.R = R # measurement noise covariance
        ukf.Q = Q # process noise covariance

        self.ukf = ukf

    def estimate(self, z: np.ndarray, u: np.ndarray, validity: np.ndarray):
        assert z.shape == (self.sensor.num_outputs,), f"{z.shape=} != {(self.sensor.num_outputs,)=}"
        assert u.shape == (self.model.num_inputs,), f"{u.shape=} != {(self.model.num_inputs,)=}"
        assert validity.shape == (self.sensor.num_outputs,), f"{validity.shape=} != {(self.sensor.num_outputs,)=}"

        self.ukf.predict(u=u)
        self.ukf.update(z)

        return self.ukf.x
