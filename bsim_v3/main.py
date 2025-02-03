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
import time
from more_itertools import powerset
from tqdm import tqdm
from lib.controllers.pure_pursuit import KinematicBicycle5StatePurePursuitController
from lib.detectors.detector import Detector, EveryEstimateDetector, LookAheadDetector, CalcValidityMetadata
from lib.detectors.utils import calc_invalid_spans
from lib.estimators.simple_ukf import SimpleUKF
from lib.fault_generators import (
    complete_failure,
    intermittent_fault,
    random_noise_fault,
    sensor_bias_fault,
    spike_fault,
)
from lib.planners.base_planner import PlannerOutput
from lib.planners.static import StaticFigureEightPlanner
from lib.planners.utils import calc_target_velocity
from lib.plants.kinematic_bicycle import \
    KinematicBicycle5StateRearWheelRefPlant
from lib.sensors.kinematic_bicycle_race_day import \
    KinematicBicycleRaceDaySensor

# check if the pdflatex binary is available
if os.system("pdflatex -v > /dev/null 2>&1") == 0:
    usetex = True
    font_serif = "Computer Modern"
else:
    print("WARNING: pdflatex not available. Not using LaTeX for plots.")
    usetex = False
    font_serif = "DejaVu Serif"

plt.rcParams.update({
    "text.usetex": usetex,
    "font.family": "serif",
    "font.serif": font_serif,
})

# %%

dt = 0.01
x0 = np.array([100, 200, 0, 0, 0])
plant = KinematicBicycle5StateRearWheelRefPlant(
    x0,
    dt,
    L=4.094, # m, taken from the racecar
    max_steer=0.59, # rad
    max_speed=107.29, # m/s
    # TODO: add max deceleration
    max_accel=3.3 * 9.81, # m/s^2
    max_steer_rate=np.inf, # rad/s
)
sensor = KinematicBicycleRaceDaySensor()
fault_generators = [
    # Sensor noise
    # random_noise_fault(0, 0, 0.1),
    # random_noise_fault(0, 1, 0.1),
    # random_noise_fault(0, 2, 0.05),
    # random_noise_fault(0, 3, 0.3),
    # random_noise_fault(0, 4, 0.3),
    # random_noise_fault(0, 5, 0.01),

    # Faults
    # sensor_bias_fault(0, 0, 10),
    # sensor_bias_fault(0, 1, 10),
    # sensor_bias_fault(18 / dt, 2, -1),
    sensor_bias_fault(18 / dt, 3, 5),
    # random_noise_fault(4 / dt, 3, 5),
    ## faults from the beginning
    # sensor_bias_fault(0, 2, -1),
    # random_noise_fault(0, 4, 0.05),
    # intermittent_fault(0, 2, 2, 10),
    # intermittent_fault(0, 3, 2, 10),
]

# Estimator
x0_hat = x0 + np.array([2, 2, 1, 0.1, 0.01]) # initial state estimate
noise_std = np.array([0.5, 0.5, 0.1, 0.5, 0.5, 0.1]) # measurement noise
P = np.diag([1,1,0.3,0.5,0.1]) # initial state covariance
R = np.diag((noise_std)**2) # measurement noise
Q = np.diag([0.1,0.1,0.01,0.1,0.001]) # process noise
estimator = SimpleUKF(plant.model, sensor, dt, x0_hat, P, R, Q)

# Planner
planner = StaticFigureEightPlanner(
    center=[0, 0],
    length=2000,
    width=1000,
    num_points=100000,
    target_velocity_fn=lambda _points, _headings, curvatures, _dK_ds_list: calc_target_velocity(
        curvatures, plant.max_speed
    ),
)

# Controller
controller = KinematicBicycle5StatePurePursuitController(
    L=plant.L,
    max_steer_rate=plant.model.max_steer_rate,
    max_accel=plant.model.max_accel,
    lookahead_fn=lambda v: 0.25 * v,
    dt=dt,
)

# Detector
N = plant.model.num_states
# N = plant.model.num_states * 2
# detector_class = Detector
detector_class = LookAheadDetector
# detector_class = EveryEstimateDetector
# Sensors 0 and 1 are protected
detector = detector_class(
    plant.model,
    sensor,
    N,
    dt,
    noise_std * 3,
    # np.array([1.5, 1.5, 0.3, 1.5, 1.5, 0.3]) * 0.01675478978252254,
    # Only consider cases where sensors 0 and 1 are valid
    # This is done because we know that sensors 0 and 1 are protected
    [s for s in powerset(range(6)) if {0, 1}.issubset(s)],
)

# Simulate the plant
x_list: list[np.ndarray] = []
z_list: list[np.ndarray] = []
validity_list: list[np.ndarray] = []
x_hat_list: list[np.ndarray] = []
u_list: list[np.ndarray] = []
plans: list[PlannerOutput] = []
controller_meta: list[dict] = []
calc_validity_meta: list[CalcValidityMetadata] = []
print("Starting simulation...")
start = time.perf_counter()
for k in tqdm(range(int(25 / dt))):
    # print(f"{k=}")

    state = plant.get_state()
    true_output = sensor.get_output(state)
    output = true_output
    for fault in fault_generators:
        output = fault(k, output)

    validity, cv_meta = detector.calc_validity()
    estimate = estimator.estimate(output, plant.u, validity)
    plan = planner.plan(estimate)
    inp, ctrl_meta = controller.step(plan, estimate)

    # We don't need inp at this time step,
    # but we will store it for the next time step 
    detector.step(output, estimate, plan, inp)

    plant.set_inputs(inp)
    plant.next()
    
    x_list.append(state)
    z_list.append(output)
    validity_list.append(validity)
    x_hat_list.append(estimate)
    u_list.append(inp)
    plans.append(plan)
    controller_meta.append(ctrl_meta)
    calc_validity_meta.append(cv_meta)

end = time.perf_counter()
print(f"Simulation complete in {end - start:.2f} s")


x = np.array(x_list)
x_hat = np.array(x_hat_list)
z = np.array(z_list)
validities = np.array(validity_list)

print("Invalid sensors (time step, sensor idx)")
print(np.array((validities != True).nonzero()).transpose())

# %%
# Plot the results
# BEV
fig = plt.figure(figsize=(7.3*1.5, 7.3*1.5))
ax = plt.subplot(321)
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
ax.grid()

# Heading
ax = plt.subplot(322)
ax.plot(np.unwrap(x[:, 2]), label="True", linestyle="--")
ax.plot(np.unwrap([x_[2] for x_ in x_hat]), label="Estimated")
ax.plot(np.unwrap([z_[2] for z_ in z]), label="Measured", linestyle=":", alpha=0.5)
ax.plot(np.unwrap([m["target_heading"] for m in controller_meta]), label="Target", alpha=0.8, zorder=-1)
for s, e in calc_invalid_spans(validities, 2):
    ax.axvspan(s, e, color="red", alpha=0.2)
ax.set_xlabel("Time step")
ax.set_ylabel("Heading [rad]")
ax.set_title("Heading")
ax.legend()
ax.grid()

# Velocity
ax = plt.subplot(323)
ax.plot(x[:, 3], label="True", linestyle="--")
ax.plot([x_[3] for x_ in x_hat], label="Estimated")
ax.plot([z_[3] for z_ in z], label="Measured (v1)", linestyle=":", alpha=0.5)
ax.plot([z_[4] for z_ in z], label="Measured (v2)", linestyle=":", alpha=0.5)
ax.plot([m["target_velocity"] for m in controller_meta], label="Target", alpha=0.8, zorder=-1)
for s, e in calc_invalid_spans(validities, 3):
    ax.axvspan(s, e, color="red", alpha=0.2)
for s, e in calc_invalid_spans(validities, 4):
    ax.axvspan(s, e, color="orange", alpha=0.2)
ax.set_xlabel("Time step")
ax.set_ylabel("Velocity [m/s]")
ax.set_title("Velocity")
ax.legend()
ax.grid()

# Steering angle
ax = plt.subplot(324)
ax.plot(x[:, 4], label="True", linestyle="--")
ax.plot([x_[4] for x_ in x_hat], label="Estimated")
ax.plot([z_[5] for z_ in z], label="Measured", linestyle=":", alpha=0.5)
ax.plot([m["target_delta"] for m in controller_meta], label="Target", alpha=0.8, zorder=-1)
for s, e in calc_invalid_spans(validities, 5):
    ax.axvspan(s, e, color="red", alpha=0.2)
ax.set_xlabel("Time step")
ax.set_ylabel("Steering angle [rad]")
ax.set_title("Steering angle")
ax.legend()
ax.grid()

# Sensor validity (horizontal stacked bar chart)
ax = plt.subplot(325)
ax.invert_yaxis()
colors = {True: "tab:green", False: "tab:red"}
for i in range(validities.shape[1]):
    for j in range(validities.shape[0]):
        ax.barh(i + 1, 1, left=j, color=colors[validities[j, i]])
ax.set_yticks(np.arange(validities.shape[1]) + 1)
ax.set_xlabel("Time step")
ax.set_ylabel("Sensor")
ax.set_title("Sensor validity")
ax.grid(axis="x")

# Detector runtime
ax = plt.subplot(326)
ax.plot([m.total_time for m in calc_validity_meta], label="Total time")
ax.plot([m.optimizer_metadata.setup_time if m.optimizer_metadata else np.nan for m in calc_validity_meta], label="Setup time")
ax.plot([m.optimizer_metadata.solve_time if m.optimizer_metadata else np.nan for m in calc_validity_meta], label="Solve time")
ax.set_ylim(0, np.max([m.total_time for m in calc_validity_meta] + [0.02]) * 1.1)
ax.set_xlabel("Time step")
ax.set_ylabel("Time [s]")
ax.set_title("Detector runtime")
ax.legend()
ax.grid()

fig.tight_layout()
fig.show()

# %%
