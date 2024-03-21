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
from tqdm import tqdm
from lib.controllers.pure_pursuit import KinematicBicycle5StatePurePursuitController
from lib.estimators.simple_ukf import SimpleUKF
from lib.fault_generators import (
    complete_failure,
    intermittent_fault,
    random_noise_fault,
    sensor_bias_fault,
    spike_fault,
)
from lib.planners.static import StaticFigureEightPlanner
from lib.planners.utils import calc_target_velocity
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
x0 = np.array([20, 50, 0, 0, 0])
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
    # sensor_bias_fault(200, 2, 1),
    # sensor_bias_fault(200, 3, -5),
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
x0_hat = x0 + np.array([2, 2, 1, 0.1, 0.01]) # initial state estimate
noise_std = np.array([0.5, 0.5, 0.1, 0.5, 0.5, 0.1]) # measurement noise
P = np.diag([1,1,0.3,0.5,0.1]) # initial state covariance
R = np.diag(noise_std**2) # measurement noise
Q = np.diag([0.1,0.1,0.01,0.1,0.001]) # process noise
estimator = SimpleUKF(plant.model, sensor, dt, x0_hat, P, R, Q)

# Planner
planner = StaticFigureEightPlanner(
    center=[0, 0],
    length=200,
    width=100,
    num_points=10000,
    target_velocity_fn=lambda _points, _headings, curvatures, _dK_ds_list: calc_target_velocity(
        curvatures, plant.max_speed
    ),
)

# Controller
controller = KinematicBicycle5StatePurePursuitController(
    L=plant.L,
    max_steer_rate=plant.model.max_steer_rate,
    max_accel=plant.model.max_accel,
    lookahead_fn=lambda v: 0.5 * v,
    dt=dt,
)

# Set the inputs
plant.set_inputs([4*9.81, 0.01])

# Simulate the plant
x = []
z = []
x_hat = []
u = []
controller_meta = []
print("Starting simulation...")
for k in tqdm(range(400)):
    plant.next()
    state = plant.get_state()
    true_output = sensor.get_output(state)
    output = true_output
    for fault in fault_generators:
        output = fault(k, output)
    estimate = estimator.estimate(output, plant.u, np.ones(sensor.num_outputs))
    plan = planner.plan(estimate)
    inp, ctrl_meta = controller.step(plan, estimate)
    plant.set_inputs(inp)
    
    x.append(state)
    z.append(output)
    x_hat.append(estimate)
    u.append(inp)
    controller_meta.append(ctrl_meta)


x = np.array(x)

# Plot the results
# BEV
fig = plt.figure(figsize=(7.3, 7.3))
ax = plt.subplot(221)
ax.plot(x[:, 0], x[:, 1], label="True", linestyle="--")
ax.plot([x_[0] for x_ in x_hat], [x_[1] for x_ in x_hat], label="Estimated")
ax.plot(
    [z_[0] for z_ in z], [z_[1] for z_ in z], label="Measured", linestyle=":", alpha=0.5
)
ax.plot(planner.base_plan.points[:, 0], planner.base_plan.points[:, 1], label="Planned", alpha=0.8, zorder=-1)
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("Trajectory")
ax.axis("equal")
ax.legend()

# Heading
ax = plt.subplot(222)
ax.plot(np.unwrap(x[:, 2]), label="True", linestyle="--")
ax.plot(np.unwrap([x_[2] for x_ in x_hat]), label="Estimated")
ax.plot(np.unwrap([z_[2] for z_ in z]), label="Measured", linestyle=":", alpha=0.5)
ax.plot(np.unwrap([m["target_heading"] for m in controller_meta]), label="Target", alpha=0.8, zorder=-1)
ax.set_xlabel("Time step")
ax.set_ylabel("Heading [rad]")
ax.set_title("Heading")
ax.legend()

# Velocity
ax = plt.subplot(223)
ax.plot(x[:, 3], label="True", linestyle="--")
ax.plot([x_[3] for x_ in x_hat], label="Estimated")
ax.plot([z_[3] for z_ in z], label="Measured (v1)", linestyle=":", alpha=0.5)
ax.plot([z_[4] for z_ in z], label="Measured (v2)", linestyle=":", alpha=0.5)
ax.plot([m["target_velocity"] for m in controller_meta], label="Target", alpha=0.8, zorder=-1)
ax.set_xlabel("Time step")
ax.set_ylabel("Velocity [m/s]")
ax.set_title("Velocity")
ax.legend()

# Steering angle
ax = plt.subplot(224)
ax.plot(x[:, 4], label="True", linestyle="--")
ax.plot([x_[4] for x_ in x_hat], label="Estimated")
ax.plot([z_[5] for z_ in z], label="Measured", linestyle=":", alpha=0.5)
ax.plot([m["target_delta"] for m in controller_meta], label="Target", alpha=0.8, zorder=-1)
ax.set_xlabel("Time step")
ax.set_ylabel("Steering angle [rad]")
ax.set_title("Steering angle")
ax.legend()

fig.tight_layout()
fig.show()
