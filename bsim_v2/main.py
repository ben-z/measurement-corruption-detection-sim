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
import matplotlib.pyplot as plt
import numpy as np
import time
from math import pi, sin, cos, atan2, sqrt
from multiprocessing import Pool
from numpy.typing import NDArray
from typing import List, Tuple, Any
from utils import (
    generate_figure_eight_approximation,
    Optimizer,
    plot_quad,
    run_simulation,
    find_corruption,
    run_experiments,
    kinematic_bicycle_model_linearize,
    kinematic_bicycle_model_desired_state_at_idx,
    kinematic_bicycle_model_normalize_output,
    calculate_segment_lengths,
)
from fault_generators import (
    sensor_bias_fault,
    intermittent_fault,
    drift_fault,
)

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
np.set_printoptions(suppress=True)


# %%

g = 9.81  # m/s^2

model_params = {
    "dt": 0.01,
    "l": 4.094, # taken from the Indy racecar
    "max_linear_acceleration": 4 * g,  # m/s^2
    "max_steering_rate": 10,  # rad/s
    "max_linear_velocity": 300 / 3.6,  # m/s
    "min_linear_velocity": 0.1,  # m/s
}

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

# Plot the path
plt.figure()
plt.plot([p[0] for p in path_points], [p[1] for p in path_points])
plt.axis("equal")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.plot(path_points[0][0], path_points[0][1], "ro")
# Arrow indicating the initial heading of the path
plt.arrow(
    path_points[0][0],
    path_points[0][1],
    200 * cos(path_headings[0]),
    200 * sin(path_headings[0]),
    width=10,
    head_width=20,
    head_length=50,
    fc="r",
    ec="r",
    zorder=10,  # Set a higher z-order to bring the arrow to the front
)
plt.legend(
    [
        plt.Line2D(range(1), range(1), color="white", marker="o", markerfacecolor="r"),
        plt.Line2D([0], [0], color="r", lw=2, linestyle=(4, [0,4,4,4]), marker=">", markersize=6),
    ],
    ["Path start", "Direction of travel"],
)
# save to file (pdf)
plt.savefig("figure-eight-path.pdf", format="pdf")
plt.title("Path")
plt.show()


# Plot the original velocity profile and the smoothed velocity profile
plt.figure()
plt.plot(np.cumsum(path_lengths), velocity_profile, label='smoothed')
plt.ylim([0, 1.1 * max(velocity_profile)])
plt.xlabel("Distance along path (m)")
plt.ylabel("Velocity (m/s)")
plt.savefig("velocity-profile.pdf", format="pdf")
plt.title('Velocity profile')
plt.plot(np.cumsum(path_lengths), velocity_profile_raw, label='original')
plt.legend()
plt.show()

# %%

# Starting position
x0 = np.array([200, 100, pi / 4, 1, 0])
# Measurement matrix
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
# Noise standard deviation  
noise_std: NDArray[np.float64] = np.array([0.5, 0.5, 0.05, 0.5, 0.01, 0.01])
# Detector window size
N = x0.shape[0]
# Optimizer used in the detector
optimizer = Optimizer(N, C.shape[0], C.shape[1])
# Run the detector in the loop
real_time_fault_tolerance = False

start = time.perf_counter()
# Run the attack detector out of the loop, after we record the data.
for attack_start_t in [10]:
    # for attack_start_t in range(10, 50, 10):
    print(f"Running simulation with attack starting at t={attack_start_t}")

    simulation_seconds = attack_start_t + 50
    num_steps = int(simulation_seconds / model_params["dt"])

    # fault = sensor_bias_fault(attack_start_t, 3, 10)
    fault = sensor_bias_fault(attack_start_t, 2, 0.5)
    # fault = drift_fault(attack_start_t, 3, -3)
    # fault = drift_fault(attack_start_t, 2, -0.05)

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
    corruption = next(find_corruption(
        output_hist,
        u_hist,
        closest_idx_hist,
        path_points,
        path_headings,
        velocity_profile,
        [C] * len(output_hist),
        N,
        500,
        optimizer,
        model_params,
        noise_std,
        model_at_idx=lambda idx: kinematic_bicycle_model_linearize(path_headings[idx], velocity_profile[idx], 0, model_params['dt'], model_params['l']),
        desired_output_fn=lambda i, idx: C @ kinematic_bicycle_model_desired_state_at_idx(idx, path_points, path_headings, velocity_profile),
        normalize_output=kinematic_bicycle_model_normalize_output,
    ))
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
