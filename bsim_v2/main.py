# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
# ---

# %%
#! echo "If you see this in the output, magic commands are correctly configured."
# Automatically reload modules
#! %load_ext autoreload
#! %autoreload 2

import os
import sys

# Disable multithreading for numpy. Must be done before importing numpy.
# Disabling because numpy is slower with multithreading for this application on machines with high single-core performance.
NUM_THREADS = "1"
os.environ["OMP_NUM_THREADS"] = NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS
os.environ["MKL_NUM_THREADS"] = NUM_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS

import cvxpy as cp
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import time
from math import pi, sin, cos, atan2, sqrt
from multiprocessing import Pool
from numpy.typing import NDArray
from typing import List, Tuple, Any
from utils import (
    get_unpack_fn,
    generate_circle_approximation,
    generate_figure_eight_approximation,
    kinematic_bicycle_model,
    PIDController,
    closest_point_idx,
    closest_point_idx_local,
    get_lookahead_idx,
    wrap_to_pi,
    clamp,
    walk_trajectory_by_durations,
    kinematic_bicycle_model_linearize,
    calc_input_effects_on_output,
    get_output_evolution_tensor,
    get_state_evolution_tensor,
    optimize_l0_v2,
    Optimizer,
    MyOptimizationCaseResult,
    MyOptimizerRes,
    format_floats,
    MAX_POOL_SIZE,
    get_solver_setup,
    plot_quad,
    estimate_state,
    run_simulation,
    find_corruption,
    run_experiments,
)
from fault_generators import (
    sensor_bias_fault,
    intermittent_fault,
    drift_fault,
)

plt.rcParams["text.usetex"] = True
np.set_printoptions(suppress=True)


# %%

g = 9.81  # m/s^2

model_params = {
    "dt": 0.01,
    "l": 0.5,
    "max_linear_acceleration": 4 * g,  # m/s^2
    "max_steering_rate": 10,  # rad/s
    "max_linear_velocity": 300 / 3.6,  # m/s
    "min_linear_velocity": 0.1,  # m/s
}


def calculate_segment_lengths(points: List[Tuple[float, float]]) -> List[float]:
    """Calculate the lengths of each segment given a list of (x, y) tuples."""
    return [
        math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        for (x1, y1), (x2, y2) in zip(points, np.roll(points, -1, axis=0))
    ]


# path_points, path_headings, path_curvatures, path_dcurvatures = generate_figure_eight_approximation([0, 0], 10, 5, 1000)
# path_points, path_headings, path_curvatures, path_dcurvatures = generate_circle_approximation([-10, 0], 10, 1000)
(
    path_points,
    path_headings,
    path_curvatures,
    path_dcurvatures,
) = generate_figure_eight_approximation([0, 0], 2000, 1000, 100000)
# path_points, path_headings, path_curvatures, path_dcurvatures = generate_figure_eight_approximation([0, 0], 1000, 500, 100000)

path_lengths = calculate_segment_lengths(path_points)

path_max_curvature = max(np.abs(path_curvatures))
path_min_turning_radius = 1 / path_max_curvature
print(
    f"Maximum curvature: {path_max_curvature:.4f} m^-1 (minimum turning radius: {path_min_turning_radius:.2f} m)"
)

side_friction_factor = (
    0.1 + path_min_turning_radius / 100
)  # unitless. Determines the maximum lateral force a car can take. 0.15 for urban, 0.1 for highway, (much) higher for racecars.
# side_friction_factor = 2

velocity_profile_raw = np.sqrt(
    side_friction_factor * g / np.clip(np.abs(path_curvatures), 1e-6, None)
)
velocity_profile = np.clip(
    velocity_profile_raw,
    model_params["min_linear_velocity"],
    model_params["max_linear_velocity"],
)

# # Plot the original velocity profile and the smoothed velocity profile
# plt.figure()
# plt.plot(velocity_profile, label='smoothed')
# plt.plot(velocity_profile_raw, label='original')
# plt.title('Velocity profile')
# plt.ylim([0, 1.1 * max(velocity_profile)])
# plt.legend()
# plt.show()


# Control the system to follow the path
x0 = np.array([200, 100, pi / 4, 1, 0])
N = x0.shape[0]
C = np.array(
    [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
    ]
)
noise_std: NDArray[np.float64] = np.array([0.5, 0.5, 0.05, 0.5, 0.01, 0.01])
optimizer = Optimizer(N, C.shape[0], C.shape[1])
real_time_fault_tolerance = False

start = time.perf_counter()
# Run the attack detector out of the loop, after we record the data.
for attack_start_t in [10]:
    # for attack_start_t in range(10, 50, 10):
    print(f"Running simulation with attack starting at t={attack_start_t}")

    simulation_seconds = attack_start_t + 50
    num_steps = int(simulation_seconds / model_params["dt"])

    # fault = sensor_bias_fault(attack_start_t, 3, 10)
    # fault = sensor_bias_fault(attack_start_t, 2, 0.5)
    # fault = drift_fault(attack_start_t, 3, -3)
    fault = drift_fault(attack_start_t, 2, -0.05)

    (
        t_hist,
        state_hist,
        output_hist,
        estimate_hist,
        u_hist,
        closest_idx_hist,
        ukf_P_hist,
    ) = run_simulation(
        x0,
        C,
        noise_std,
        num_steps,
        N,
        path_points,
        path_headings,
        path_curvatures,
        path_dcurvatures,
        velocity_profile,
        optimizer,
        model_params,
        fault,
        real_time_fault_tolerance,
    )

    generate_gif = plot_quad(
        t_hist,
        state_hist,
        output_hist,
        estimate_hist,
        u_hist,
        closest_idx_hist,
        ukf_P_hist,
        path_points,
        path_headings,
        velocity_profile,
        model_params,
    )
    plt.show()

    # Post-analysis
    print("Finding corruption")
    corruption = find_corruption(
        output_hist,
        u_hist,
        closest_idx_hist,
        path_points,
        path_headings,
        velocity_profile,
        [C] * N,
        N,
        500,
        optimizer,
        model_params,
        noise_std,
    )
    if corruption is None:
        print("No corruption found")
    else:
        det_delay = corruption["t"] - attack_start_t
        print(
            f"Detected corruption {det_delay:.2f}s after injection at k={corruption['k']} (t={corruption['t']:.2f}s)"
            + f", K={corruption['K']}"
            + f", estimator/optimizer/solve time: {corruption['metadata']['total_time']:.4f}/{corruption['metadata']['optimizer_time']:.4f}/{corruption['optimizer_metadata']['solve_time']:.4f}s"
        )

end = time.perf_counter()
print(f"Simulation took {end - start:.2f} seconds")


# %%

fault_specs = []
num_passes = 30 * 3
file_name = "./exp/test-drift-sensors-4-5.jsonl"
exp_name = "drift-fault-sweep-sensors-4-5"

# Corrupt the velocity sensor
# for bias in np.arange(-5, 5, 0.05):
#     for start_t in [10, 15, 20, 25, 30, 35, 40]:
#         fault_specs.append({"fn": "sensor_bias_fault", "kwargs": {"start_t": start_t, "sensor_idx": 3, "bias": bias}})

# # Corrupt the heading sensor
# for bias in np.arange(-0.5, 0.5, 0.005):
#     for start_t in [10, 15, 20, 25, 30, 35, 40]:
#         fault_specs.append({"fn": "sensor_bias_fault", "kwargs": {"start_t": start_t, "sensor_idx": 2, "bias": bias}})

# Inject fault at different points of the simulation
for start_t in [10, 15, 20, 25, 30, 35, 40]:
    #############################################
    # Bias Faults
    #############################################

    # for bias in np.arange(-np.pi/4, np.pi + sys.float_info.epsilon, 0.005):
    #     fault_specs.append(
    #         {
    #             "fn": "sensor_bias_fault",
    #             "kwargs": {"start_t": start_t, "sensor_idx": 2, "bias": bias},
    #         }
    #     )
    # for bias in np.arange(-np.pi/4, np.pi + sys.float_info.epsilon, 0.005):
    #     fault_specs.append(
    #         {
    #             "fn": "sensor_bias_fault",
    #             "kwargs": {"start_t": start_t, "sensor_idx": 4, "bias": bias},
    #         }
    #     )
    # for bias in np.arange(-np.pi/4, np.pi + sys.float_info.epsilon, 0.005):
    #     fault_specs.append(
    #         {
    #             "fn": "sensor_bias_fault",
    #             "kwargs": {"start_t": start_t, "sensor_idx": 5, "bias": bias},
    #         }
    #     )

    #############################################
    # Spike Faults
    #############################################

    # for spike_value in np.arange(-20, 20 + sys.float_info.epsilon, 0.5):
    #     for duration in np.arange(0.1, 2 + sys.float_info.epsilon, 0.1):
    #         fault_specs.append(
    #             {
    #                 "fn": "spike_fault",
    #                 "kwargs": {
    #                     "start_t": start_t,
    #                     "sensor_idx": 3,
    #                     "spike_value": spike_value,
    #                     "duration": duration,
    #                 },
    #             }
    #         )

    # for spike_value in np.arange(-np.pi, np.pi + sys.float_info.epsilon, 0.1):
    #     for duration in np.arange(0.1, 2 + sys.float_info.epsilon, 0.1):
    #         fault_specs.append(
    #             {
    #                 "fn": "spike_fault",
    #                 "kwargs": {
    #                     "start_t": start_t,
    #                     "sensor_idx": 2,
    #                     "spike_value": spike_value,
    #                     "duration": duration,
    #                 },
    #             }
    #         )

    # for spike_value in np.arange(-np.pi/2, np.pi/2 + sys.float_info.epsilon, 0.1):
    #     for duration in np.arange(0.1, 2 + sys.float_info.epsilon, 0.1):
    #         fault_specs.append(
    #             {
    #                 "fn": "spike_fault",
    #                 "kwargs": {
    #                     "start_t": start_t,
    #                     "sensor_idx": 4,
    #                     "spike_value": spike_value,
    #                     "duration": duration,
    #                 },
    #             }
    #         )

    # for spike_value in np.arange(-np.pi/2, np.pi/2 + sys.float_info.epsilon, 0.1):
    #     for duration in np.arange(0.1, 2 + sys.float_info.epsilon, 0.1):
    #         fault_specs.append(
    #             {
    #                 "fn": "spike_fault",
    #                 "kwargs": {
    #                     "start_t": start_t,
    #                     "sensor_idx": 5,
    #                     "spike_value": spike_value,
    #                     "duration": duration,
    #                 },
    #             }
    #         )

    #############################################
    # Noise Faults
    #############################################

    # for noise_level in np.arange(0, 20 + sys.float_info.epsilon, 0.5):
    #     fault_specs.append(
    #         {
    #             "fn": "random_noise_fault",
    #             "kwargs": {
    #                 "start_t": start_t,
    #                 "sensor_idx": 3,
    #                 "noise_level": noise_level,
    #             },
    #         }
    #     )

    # for noise_level in np.arange(0, np.pi + sys.float_info.epsilon, 0.01):
    #     fault_specs.append(
    #         {
    #             "fn": "random_noise_fault",
    #             "kwargs": {
    #                 "start_t": start_t,
    #                 "sensor_idx": 2,
    #                 "noise_level": noise_level,
    #             },
    #         }
    #     )

    # for noise_level in np.arange(0, np.pi / 4 + sys.float_info.epsilon, 0.01):
    #     fault_specs.append(
    #         {
    #             "fn": "random_noise_fault",
    #             "kwargs": {
    #                 "start_t": start_t,
    #                 "sensor_idx": 4,
    #                 "noise_level": noise_level,
    #             },
    #         }
    #     )

    # for noise_level in np.arange(0, np.pi / 4 + sys.float_info.epsilon, 0.01):
    #     fault_specs.append(
    #         {
    #             "fn": "random_noise_fault",
    #             "kwargs": {
    #                 "start_t": start_t,
    #                 "sensor_idx": 5,
    #                 "noise_level": noise_level,
    #             },
    #         }
    #     )

    #############################################
    # Drift Faults
    #############################################

    # for drift_rate in np.arange(-20, 20 + sys.float_info.epsilon, 0.5):
    #     fault_specs.append(
    #         {
    #             "fn": "drift_fault",
    #             "kwargs": {
    #                 "start_t": start_t,
    #                 "sensor_idx": 3,
    #                 "drift_rate": drift_rate,
    #             },
    #         }
    #     )

    # for drift_rate in np.arange(-np.pi, np.pi + sys.float_info.epsilon, 0.005):
    #     fault_specs.append(
    #         {
    #             "fn": "drift_fault",
    #             "kwargs": {
    #                 "start_t": start_t,
    #                 "sensor_idx": 2,
    #                 "drift_rate": drift_rate,
    #             },
    #         }
    #     )
    
    for drift_rate in np.arange(-np.pi/16, np.pi/16 + sys.float_info.epsilon, 0.005):
        fault_specs.append(
            {
                "fn": "drift_fault",
                "kwargs": {
                    "start_t": start_t,
                    "sensor_idx": 4,
                    "drift_rate": drift_rate,
                },
            }
        )
    
    for drift_rate in np.arange(-np.pi/16, np.pi/16 + sys.float_info.epsilon, 0.005):
        fault_specs.append(
            {
                "fn": "drift_fault",
                "kwargs": {
                    "start_t": start_t,
                    "sensor_idx": 5,
                    "drift_rate": drift_rate,
                },
            }
        )

print(f"Experiment name: {exp_name}")
print(f"Running {len(fault_specs)} experiments for {num_passes} passes. Total: {len(fault_specs) * num_passes} experiments")
print(f"Saving to {file_name}")

# For testing
# sys.exit(0)

simulation_seconds = 50
num_steps = int(simulation_seconds / model_params["dt"])

for i in range(1, num_passes+1):
    print(f"Running experiment pass {i}/{num_passes}")
    start = time.perf_counter()
    run_experiments(
        file_name,
        x0,
        C,
        noise_std,
        num_steps,
        N,
        path_points,
        path_headings,
        path_curvatures,
        path_dcurvatures,
        velocity_profile,
        optimizer,
        model_params,
        real_time_fault_tolerance,
        # Fault specification
        fault_specs,
        extra_output_metadata={
            "exp_name": exp_name,
            "exp_pass": i,
        },
    )
    print(f"Experiment pass {i}/{num_passes} took {time.perf_counter() - start:.2f} seconds")


# %%
# Retroactively calculate from data the modelling error and the noise
# errors = []

# # Arbitrarily ignore the first 500 data points because they are not at steady state
# for i in range(max(N, 500), len(output_hist)+1):
#     solver_setup = get_solver_setup(output_hist[i-N:i], u_hist[i-N:i-1], closest_idx_hist[i-N:i-1], path_points, path_headings, velocity_profile, [C]*N, dt=model_params['dt'], l=model_params['l'])
#     error = solver_setup['Phi'] @ solver_setup['output_hist_no_input_effects'][0][:5] - solver_setup['Y']
#     error[:,2] = wrap_to_pi(error[:,2])
#     error[:,4] = wrap_to_pi(error[:,4])
#     error[:,5] = wrap_to_pi(error[:,5])
#     errors.append(error)

# errors_tensor = np.array(errors)

# max_errors_over_time = abs(errors_tensor).max(axis=1)
# x = np.arange(len(max_errors_over_time))
# for i in range(6):
#     plt.plot(x, max_errors_over_time[:,i], ".-", label=f"sensor {i}")
# plt.xlabel("time step")
# plt.ylabel("max error")
# plt.title("Max error per sensor over time")
# plt.legend()
# plt.show()

# print("Max error per sensor and per time step in an interval")
# print(abs(errors_tensor).max(axis=0))

# print("95th percentile error per sensor per time step in an interval")
# print(np.percentile(abs(errors_tensor), 95, axis=0))

# print("90th percentile error per sensor per time step in an interval")
# print(np.percentile(abs(errors_tensor), 90, axis=0))

# generate_gif()

# %%

# # Plot velocity on top of target velocity
# plt.figure()
# plt.plot([velocity_profile[idx] for idx in closest_idx_hist], label=r"$v_d$") # target velocity
# plt.plot([p[3] for p in state_hist], label=r"$v$") # velocity
# plt.title("Velocity")
# plt.legend()
# plt.show()

# # Plot heading on top of target heading
# plt.figure()
# plt.plot([path_headings[idx] for idx in closest_idx_hist], label=r"$\theta_d$") # target heading
# plt.plot([wrap_to_pi(p[2]) for p in state_hist], label=r"$\theta$") # heading
# plt.title('Heading')
# plt.legend()
# plt.show()

# # Plot control signals
# fig, axs = plt.subplots(2, 1, sharex=True)
# axs[0].plot([u[0] for u in u_hist], label=r"$a$") # a
# axs[0].set_title('Control signals')
# axs[0].legend()
# axs[1].plot([u[1] for u in u_hist], label=r"$\dot{\delta}$") # delta_dot
# axs[1].legend()
# plt.show()
# %%
