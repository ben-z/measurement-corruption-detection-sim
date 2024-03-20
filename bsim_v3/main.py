# %%
#! echo "If you see this in the output, magic commands are correctly configured."
# Automatically reload modules
#! %load_ext autoreload
#! %autoreload 2

import os

# Disable multithreading for numpy. Must be done before importing numpy.
# Disabling because numpy is slower with multithreading for this application on machines with high single-core performance.
NUM_THREADS = "1"
os.environ["OMP_NUM_THREADS"] = NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS
os.environ["MKL_NUM_THREADS"] = NUM_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS

import matplotlib.pyplot as plt
import numpy as np
from lib.estimators.simple_ukf import SimpleUKF
from lib.fault_generators import (
    complete_failure,
    intermittent_fault,
    random_noise_fault,
    sensor_bias_fault,
    spike_fault,
)
from lib.plants.kinematic_bicycle import \
    KinematicBicycle5StateRearWheelRefPlant
from lib.sensors.kinematic_bicycle_race_day import \
    KinematicBicycleRaceDaySensor

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
})

dt = 0.1
x0 = np.array([0, 0, 0, 0, 0])
plant = KinematicBicycle5StateRearWheelRefPlant(
    x0,
    dt,
    L=4.094, # m, taken from the racecar
    max_steer=0.5, # rad
    max_speed=300 / 3.6, # m/s
    max_accel=4 * 9.81, # m/s^2
    max_steer_rate=10, # rad/s
)
sensor = KinematicBicycleRaceDaySensor()
fault_generators = [
    # sensor_bias_fault(0, 0, 10),
    # sensor_bias_fault(0, 1, 10),
    # sensor_bias_fault(20, 2, 1),
    random_noise_fault(0, 0, 0.1),
    random_noise_fault(0, 1, 0.1),
    random_noise_fault(0, 2, 0.05),
    random_noise_fault(0, 3, 0.3),
    random_noise_fault(0, 4, 0.3),
    random_noise_fault(0, 5, 0.01),
    # random_noise_fault(0, 3, 0.5),
    # random_noise_fault(0, 4, 0.05),
    # intermittent_fault(0, 2, 2, 10),
    # intermittent_fault(0, 3, 2, 10),
]

# Estimator
x0_hat = x0 + np.array([2, 2, 1, 0.1, 0.01])
noise_std = np.array([0.5, 0.5, 0.1, 0.5, 0.5, 0.1])
P = np.diag([1,1,0.3,0.5,0.1]) # initial state covariance
R = np.diag(noise_std**2) # measurement noise
Q = np.diag([0.1,0.1,0.01,0.1,0.001]) # process noise
estimator = SimpleUKF(plant.model, sensor, dt, x0_hat, P, R, Q)

# Set the inputs
plant.set_inputs([4*9.81, 0.01])

# Simulate the plant
x = []
z = []
x_hat = []
for k in range(100):
    plant.next()
    state = plant.get_state()
    true_output = sensor.get_output(state)
    output = true_output
    for fault in fault_generators:
        output = fault(k, output)
    estimate = estimator.estimate(output, plant.u, np.ones(sensor.num_outputs))

    x.append(state)
    z.append(output)
    x_hat.append(estimate)

x = np.array(x)

# Plot the results
# BEV
fig = plt.figure(figsize=(7.3, 7.3))
ax = plt.subplot(221)
ax.plot(x[:, 0], x[:, 1], label="True", linestyle="--")
ax.plot([x_[0] for x_ in x_hat], [x_[1] for x_ in x_hat], label="Estimated")
ax.plot([z_[0] for z_ in z], [z_[1] for z_ in z], label="Measured", linestyle=":")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("Trajectory")
ax.axis("equal")
ax.legend()

# Heading
ax = plt.subplot(222)
ax.plot(np.unwrap(x[:, 2]), label="True", linestyle="--")
ax.plot(np.unwrap([x_[2] for x_ in x_hat]), label="Estimated")
ax.plot(np.unwrap([z_[2] for z_ in z]), label="Measured", linestyle=":")
ax.set_xlabel("Time step")
ax.set_ylabel("Heading [rad]")
ax.set_title("Heading")
ax.legend()

# Velocity
ax = plt.subplot(223)
ax.plot(x[:, 3], label="True", linestyle="--")
ax.plot([x_[3] for x_ in x_hat], label="Estimated")
ax.plot([z_[3] for z_ in z], label="Measured (v1)", linestyle=":")
ax.plot([z_[4] for z_ in z], label="Measured (v2)", linestyle=":")
ax.set_xlabel("Time step")
ax.set_ylabel("Velocity [m/s]")
ax.set_title("Velocity")
ax.legend()

# Steering angle
ax = plt.subplot(224)
ax.plot(x[:, 4], label="True", linestyle="--")
ax.plot([x_[4] for x_ in x_hat], label="Estimated")
ax.plot([z_[5] for z_ in z], label="Measured", linestyle=":")
ax.set_xlabel("Time step")
ax.set_ylabel("Steering angle [rad]")
ax.set_title("Steering angle")
ax.legend()

fig.tight_layout()
fig.show()
