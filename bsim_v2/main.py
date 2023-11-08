# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
# ---

#%%
#! echo "If you see this in the output, magic commands are correctly configured."
# Automatically reload modules
#! %load_ext autoreload
#! %autoreload 2

import cvxpy as cp
import os
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
)

plt.rcParams['text.usetex'] = True
np.set_printoptions(suppress=True)


#%%

g = 9.81 # m/s^2

model_params = {
    'dt': 0.01,
    'l': 0.5,
    'max_linear_acceleration': 4*g, # m/s^2
    'max_steering_rate': 10, # rad/s
    'max_linear_velocity': 300/3.6, # m/s
    'min_linear_velocity': 0.1 # m/s
}

def limit_difference(arr, max_diff):
    # Calculate the differences
    differences = np.clip(np.diff(arr), -max_diff, max_diff)

    # Reconstruct the array
    limited_arr = np.cumsum(np.insert(differences, 0, arr[0]))

    return limited_arr

def smooth_velocity_profile(velocity_profile, path_lengths, max_acc):
    """
    Smooths the velocity profile.
    """
    max_acc = np.broadcast_to(max_acc, len(velocity_profile))

    # roll the arrays so that the smallest velocity is first
    min_vel_idx = np.argmin(velocity_profile)
    velocity_profile = np.roll(velocity_profile, -min_vel_idx)
    max_acc = np.roll(max_acc, -min_vel_idx)
    path_lengths = np.roll(path_lengths, -min_vel_idx)

    for i in range(len(velocity_profile)):
        v0 = velocity_profile[i]
        vf = velocity_profile[(i+1) % len(velocity_profile)]
        d = path_lengths[i]
    
        # Clamp the acceleration
        a = np.clip((vf**2 - v0**2) / (2 * d), -max_acc[i], max_acc[i])
        velocity_profile[(i+1) % len(velocity_profile)] = np.sqrt(v0**2 + 2 * a * d)
        
    # roll the array back
    velocity_profile = np.roll(velocity_profile, min_vel_idx)

    return velocity_profile

def calculate_segment_lengths(points: List[Tuple[float, float]]) -> List[float]:
    """Calculate the lengths of each segment given a list of (x, y) tuples."""
    return [math.sqrt((x2 - x1)**2 + (y2 - y1)**2) for (x1, y1), (x2, y2) in zip(points, np.roll(points, -1, axis=0))]

# path_points, path_headings, path_curvatures, path_dcurvatures = generate_figure_eight_approximation([0, 0], 10, 5, 1000)
# path_points, path_headings, path_curvatures, path_dcurvatures = generate_circle_approximation([-10, 0], 10, 1000)
path_points, path_headings, path_curvatures, path_dcurvatures = generate_figure_eight_approximation([0, 0], 2000, 1000, 100000)
# path_points, path_headings, path_curvatures, path_dcurvatures = generate_figure_eight_approximation([0, 0], 1000, 500, 100000)

path_lengths = calculate_segment_lengths(path_points)

path_max_curvature = max(np.abs(path_curvatures))
path_min_turning_radius = 1/path_max_curvature
print(f"Maximum curvature: {path_max_curvature:.4f} m^-1 (minimum turning radius: {path_min_turning_radius:.2f} m)")

side_friction_factor = 0.1 + path_min_turning_radius/100 # unitless. Determines the maximum lateral force a car can take. 0.15 for urban, 0.1 for highway, (much) higher for racecars.
# side_friction_factor = 2

velocity_profile_raw = np.sqrt(side_friction_factor * g / np.clip(np.abs(path_curvatures), 1e-6, None))
velocity_profile = np.clip(velocity_profile_raw, model_params["min_linear_velocity"], model_params["max_linear_velocity"])
# velocity_profile = smooth_velocity_profile(velocity_profile, path_lengths, max_linear_acceleration)
velocity_profile = np.clip(velocity_profile, model_params["min_linear_velocity"], model_params["max_linear_velocity"])

# # Plot the original velocity profile and the smoothed velocity profile
# plt.figure()
# plt.plot(velocity_profile, label='smoothed')
# plt.plot(velocity_profile_raw, label='original')
# plt.title('Velocity profile')
# plt.ylim([0, 1.1 * max(velocity_profile)])
# plt.legend()
# plt.show()


# Control the bicycle to follow the path
simulation_seconds = 15
num_steps = int(simulation_seconds / model_params['dt'])
N = 10
x0 = np.array([200,100, pi/4, 1, 0])
C = np.array([
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1],
])
noise_std: NDArray[np.float64] = np.array([0.5,0.5,0.05,0.5,0.01,0.01])
optimizer = Optimizer(N,C.shape[0],C.shape[1])

t_hist, state_hist, output_hist, estimate_hist, u_hist, closest_idx_hist, ukf_P_hist = run_simulation(x0, C, noise_std, num_steps, N, path_points, path_headings, path_curvatures, path_dcurvatures, velocity_profile, optimizer, model_params)

# # Plot state_hist on top of path_points
# plt.figure()
# plt.plot([p[0] for p in path_points], [p[1] for p in path_points], '.', label='path')
# plt.plot([p[0] for p in state_hist], [p[1] for p in state_hist], '.', label='ego')
# plt.axis('equal')
# plt.title('BEV')
# plt.legend()
# plt.show()

# %%

generate_gif = plot_quad(t_hist, state_hist, output_hist, estimate_hist, u_hist, closest_idx_hist, ukf_P_hist, path_points, path_headings, velocity_profile, model_params)
plt.show()

#%%
# Post-analysis

def estimate_state_unpack(args):
    return estimate_state(*args)


def find_corruption(output_hist, input_hist, estimate_hist, closest_idx_hist, path_points, path_headings, velocity_profile, Cs, N, starting_k, optimizer, model_params):
    # Run the solver from the beginning until we detect the corruption
    ks = range(max(N,starting_k), len(output_hist)+1)
    args_iterable = ((None, output_hist[k-N:k], input_hist[k-N:k-1], estimate_hist[k-N:k-1], closest_idx_hist[k-N:k-1], path_points, path_headings, velocity_profile, Cs, model_params['dt'], model_params['l'], N, True, optimizer, noise_std) for k in ks)

    pool = None
    res_iter = map(estimate_state_unpack, args_iterable)
    # Optional parallelization
    # pool = Pool(min(MAX_POOL_SIZE, os.cpu_count() or 1))
    # pool_chunksize = int(1/model_params['dt']) # set chunksize to be a deterministic number of seconds
    # res_iter = pool.imap(estimate_state_unpack, args_iterable, chunksize=pool_chunksize)

    try:
        # Run the solver from the beginning until we detect the corruption
        for k, (estm_res, optimizer_res, metadata) in zip(ks, res_iter):
            # print(f"{k=} (t={k*model_params['dt']:.2f}s)")
            if optimizer_res is None:
                continue

            soln, solns, optimizer_metadata = optimizer_res
            if soln is None:
                continue

            x0_hat, prob, soln_metadata = soln
            if len(soln_metadata['K']) > 0:
                print(f"Found corruption at {k=} (t={k*model_params['dt']:.2f}s), K={soln_metadata['K']}, estimator/optimizer/solve time: {metadata['total_time']:.4f}/{metadata['optimizer_time']:.4f}/{optimizer_metadata['solve_time']:.4f}")
                break
        else:
            print("No corruption detected")
    finally:
        if pool:
            pool.terminate()
            pool.join()

find_corruption(output_hist, u_hist, estimate_hist, closest_idx_hist, path_points, path_headings, velocity_profile, [C]*N, N, 500, optimizer, model_params)


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

#%%

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
