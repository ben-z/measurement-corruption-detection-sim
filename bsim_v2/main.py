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
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import time
from math import pi, sin, cos, atan2, sqrt
from typing import List, Tuple
from utils import (
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
    optimize_l0,
    format_floats,
)
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
np.set_printoptions(suppress=True)


#%%

model_params = {
    'dt': 0.01,
    'l': 0.5,
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

g = 9.81 # m/s^2
side_friction_factor = 0.1 + path_min_turning_radius/100 # unitless. Determines the maximum lateral force a car can take. 0.15 for urban, 0.1 for highway, (much) higher for racecars.
# side_friction_factor = 2
max_linear_acceleration = 4*g # m/s^2
max_steering_rate = 10 # rad/s
max_linear_velocity = 300/3.6 # m/s
min_linear_velocity = 0.1 # m/s
def get_lookahead_distance(v):
    return max(v * 0.5, 0.5)

velocity_profile_raw = np.sqrt(side_friction_factor * g / np.clip(np.abs(path_curvatures), 1e-6, None))
velocity_profile = np.clip(velocity_profile_raw, min_linear_velocity, max_linear_velocity)
# velocity_profile = smooth_velocity_profile(velocity_profile, path_lengths, max_linear_acceleration)
velocity_profile = np.clip(velocity_profile, min_linear_velocity, max_linear_velocity)

# # Plot the original velocity profile and the smoothed velocity profile
# plt.figure()
# plt.plot(velocity_profile, label='smoothed')
# plt.plot(velocity_profile_raw, label='original')
# plt.title('Velocity profile')
# plt.ylim([0, 1.1 * max(velocity_profile)])
# plt.legend()
# plt.show()

# Plot the acceleration profile
plt.figure()

def get_solver_setup(output_hist, input_hist, closest_idx_hist, path_points, path_headings, velocities, Cs, dt, l):
    """
    Returns the precomputed data needed to solve for corrupted sensors.
    Parameters:
        output_hist: list[float] - a list of N outputs
        input_hist: list[float] - a list of N-1 inputs
        closest_idx_hist: list[int] - a list of N-1 closest indices
        path_points: list[tuple[float,float]] - a list of points on the path
        velocities: list[float] - a list of velocities at each point on the path
        dt: float - the time step
        l: float - the distance between the front and rear axles
    """
    N = len(output_hist)
    assert len(input_hist) == N-1, 'input_hist must be one shorter than output_hist'
    assert len(closest_idx_hist) == N-1, 'closest_idx_hist must be one shorter than output_hist'
    assert len(Cs) == N, 'Cs must be N long'

    x0_hat_closest_idx = closest_idx_hist[0]

    # Use walk_trajectory_by_durations to get the desired path indices at each time step
    desired_path_indices = [x0_hat_closest_idx] + walk_trajectory_by_durations(path_points, velocities, x0_hat_closest_idx, [dt]*(N-1))

    # Generate the As and Bs for each time step
    models = [kinematic_bicycle_model_linearize(path_headings[idx], velocities[idx], 0, dt, l) for idx in desired_path_indices]
    As = list(m[0] for m in models[:N-1])
    Bs = list(m[1] for m in models[:N-1])
    
    # List of desired outputs at the linearization points
    desired_trajectory = [Cs[i] @ np.array([path_points[idx][0], path_points[idx][1], path_headings[idx], velocities[idx], 0]) for i, idx in enumerate(desired_path_indices)]
    
    # Calculate the effects on outputs due to inputs, then subtract them from the outputs.
    # Also subtract the desired outputs to get deviations.
    input_effects = calc_input_effects_on_output(As, Bs, Cs, input_hist)
    output_hist_no_input_effects = [output - input_effect - desired_output for output, input_effect, desired_output in zip(output_hist, input_effects, desired_trajectory)]
    # normalize angular measurements
    for o in output_hist_no_input_effects:
        o[2] = wrap_to_pi(o[2])
        o[4] = wrap_to_pi(o[4])
        o[5] = wrap_to_pi(o[5])

    # Solve for corrupted sensors
    Y = np.array(output_hist_no_input_effects)
    Phi = get_output_evolution_tensor(Cs, get_state_evolution_tensor(As))

    return {
        'Y': Y,
        'Phi': Phi,
        'output_hist_no_input_effects': output_hist_no_input_effects,
    }


def estimate_state(kf, output_hist, input_hist, estimate_hist, closest_idx_hist, path_points, path_headings, velocities, Cs, dt, l, N, enable_fault_tolerance=True):
    """
    Estimates the state of the vehicle.
    Parameters:
        kf: Optional[filterpy.kalman.UnscentedKalmanFilter] - the filter to use. If None, simply average the sensors.
        output_hist: list[float] - a list of N outputs
        input_hist: list[float] - a list of N-1 inputs
        estimate_hist: list[float] - a list of N-1 estimates
        path_points: list[tuple[float,float]] - a list of points on the path
        velocities: list[float] - a list of velocities at each point on the path
        dt: float - the time step
    Returns:
        (x_hat, y_hat, theta_hat, v_hat, delta_hat): tuple[float,float,float,float,float] - the estimated state
    """
    if kf is not None:
        if len(input_hist) == 0:
            kf.predict(u=np.array([0,0]))
        else:
            kf.predict(u=input_hist[-1])

        # print(kf.x)
        # print(kf.P)

    # This is the latest output
    output = output_hist[-1]
    assert len(output) == 6, 'This function only works with output (x,y,theta,v,delta1,delta2)'
    if kf is not None:
        kf.update(output)

        x_hat, y_hat, theta_hat, v_hat, delta_hat = kf.x
    else:
        # State estimator
        x_hat = output[0]
        y_hat = output[1]
        theta_hat = output[2]
        v_hat = output[3]
        delta_hat = np.mean([output[4], output[5]])

    if not enable_fault_tolerance:
        return x_hat,y_hat,theta_hat,v_hat,delta_hat

    ###########################################################################
    # Fault detection
    ###########################################################################
    if len(output_hist) < N:
        print(f"Insufficient data to solve for corrupt sensors. Need {N} outputs, only have {len(output_hist)}")
        return x_hat,y_hat,theta_hat,v_hat,delta_hat

    assert len(input_hist) == N-1, 'input_hist must be one shorter than output_hist'
    assert len(estimate_hist) == N-1, 'estimate_hist must be one shorter than output_hist'
    assert len(output_hist) == N, 'output_hist must be N long'

    # Use the beginning of the time window and the path to generate the As
    x0_hat = estimate_hist[0]
    x0_hat_closest_idx = closest_idx_hist[0]

    # Safe guard to prevent solving when we are too far away from the desired state
    # We are using the oldest estimate in the window to do this because it is the least
    # likely to be corrupted
    x_err = path_points[x0_hat_closest_idx][0] - x0_hat[0]
    y_err = path_points[x0_hat_closest_idx][1] - x0_hat[1]
    theta_err = wrap_to_pi(path_headings[x0_hat_closest_idx] - x0_hat[2])
    v_err = velocities[x0_hat_closest_idx] - x0_hat[3]
    delta_err = -x0_hat[4] # we use linear inerpolation, so the desrd delta is always 0
    # if abs(x_err) > 1.0 \
    #         or abs(y_err) > 1.0 \
    #         or abs(theta_err) > 0.1 \
    #         or abs(v_err) > 3.0 \
    #         or abs(delta_err) > 0.1:
    #     print(f"Too far from desired state. x_err: {x_err:.2f}, y_err: {y_err:.2f}, theta_err: {theta_err:.2f}, v_err: {v_err:.2f}, delta_err: {delta_err:.2f}. Not solving.")
    #     return x_hat,y_hat,theta_hat,v_hat,delta_hat

    setup = get_solver_setup(output_hist, input_hist, closest_idx_hist, path_points, path_headings, velocities, Cs, dt, l)

    # eps = np.array([0.1]*2+[1e-2]+[1e-3]*3)
    eps = np.array([1.5]*2+[0.3]+[1.5]+[0.05]*2)

    start = time.perf_counter()
    x0_hat, prob, metadata, solns = optimize_l0(setup['Phi'], setup['Y'], eps=eps, solver_args={'solver': cp.CLARABEL})
    end = time.perf_counter()
    assert metadata is not None, 'Optimization failed'
    for soln in solns:
        x, p, m = soln
        print(p.status, f"v: {p.value:.4f}", format_floats(m, 4), f"{p.solve_time=:.4f}, {p.compilation_time=:.4f}", "x:", format_floats(x.value, 4))
    print(f"Total/real solve time (s): {sum(m['solve_time'] for _, _, m in solns):.2f}/{end-start:.2f}")
    print("K: ", metadata['K'])

    return x_hat,y_hat,theta_hat,v_hat,delta_hat

from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

def subtract_states(x1, x2):
    return np.array([
        x1[0] - x2[0],
        x1[1] - x2[1],
        wrap_to_pi(x1[2] - x2[2]),
        x1[3] - x2[3],
        wrap_to_pi(x1[4] - x2[4])
    ])

def subtract_outputs(y1, y2):
    return np.array([
        y1[0] - y2[0],
        y1[1] - y2[1],
        wrap_to_pi(y1[2] - y2[2]),
        y1[3] - y2[3],
        wrap_to_pi(y1[4] - y2[4]),
        wrap_to_pi(y1[5] - y2[5])
    ])

# TODO: verify the implementations of state_mean and z_mean
def state_mean(sigmas, Wm):
    x = np.zeros(5)
    sum_sin = np.sum(np.dot(np.sin(sigmas[:,2]), Wm))
    sum_cos = np.sum(np.dot(np.cos(sigmas[:,2]), Wm))
    x[0] = np.dot(sigmas[:,0], Wm)
    x[1] = np.dot(sigmas[:,1], Wm)
    x[2] = atan2(sum_sin, sum_cos)
    x[3] = np.dot(sigmas[:,3], Wm)
    x[4] = np.dot(sigmas[:,4], Wm)
    return x

def z_mean(sigmas, Wm):
    z = np.zeros(6)
    sum_sin = np.sum(np.dot(np.sin(sigmas[:,2]), Wm))
    sum_cos = np.sum(np.dot(np.cos(sigmas[:,2]), Wm))
    z[0] = np.dot(sigmas[:,0], Wm)
    z[1] = np.dot(sigmas[:,1], Wm)
    z[2] = atan2(sum_sin, sum_cos)
    z[3] = np.dot(sigmas[:,3], Wm)
    z[4] = np.dot(sigmas[:,4], Wm)
    z[5] = np.dot(sigmas[:,5], Wm)
    return z

# Control the bicycle to follow the path
simulation_seconds = 30
num_steps = int(simulation_seconds / model_params['dt'])
x0 = np.array([200,100, pi/4, 1, 0])
state = x0
N = 10
C = np.array([
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1],
])
# TODO: tune the sigma points
ukf_sigma_points = MerweScaledSigmaPoints(n=C.shape[1], alpha=0.3, beta=2, kappa=3-C.shape[1], subtract=subtract_states)
ukf = UnscentedKalmanFilter(
    dim_x=C.shape[1],
    dim_z=C.shape[0],
    dt=model_params['dt'],
    fx=lambda x, _dt, u: kinematic_bicycle_model(x, u, model_params),
    hx=lambda x: C @ x,
    points=ukf_sigma_points,
    residual_x=subtract_states,
    residual_z=subtract_outputs,
    x_mean_fn=state_mean,
    z_mean_fn=z_mean,
)
ukf.x = x0
ukf.P = np.diag([1,1,0.3,0.5,0.1]) # initial state covariance
ukf.R = np.diag([0.2,0.2,0.02,0.5,0.0001,0.0001]) # measurement noise
ukf.Q = np.diag([0.1,0.1,0.01,0.1,0.001]) # process noise
state_hist = []
output_hist = []
estimate_hist = []
u_hist = []
closest_idx_hist = []
ukf_P_hist = []
a_controller = PIDController(2, 0, 0, model_params['dt'])
delta_dot_controller = PIDController(5, 0, 0, model_params['dt'])
prev_closest_idx = None
for i in range(num_steps):
    state_hist.append(state)

    # measurement
    output = C @ state
    # Add noise
    output += np.random.normal(0, [0.1,0.1,0.02,0.1,0.0001,0.0001])

    # Add attack
    if i * model_params['dt'] > 10:
        pass
        # output[2] += 1
        # output[3] += 5
        # output[4] += 0.1
    output_hist.append(output)

    # fault-tolerant estimator
    x_hat, y_hat, theta_hat, v_hat, delta_hat = estimate_state(ukf, output_hist[-N:], u_hist[-(N-1):], estimate_hist[-(N-1):], closest_idx_hist[-(N-1):], path_points, path_headings, velocity_profile, [C]*N, dt=model_params['dt'], l=model_params['l'], N=N, enable_fault_tolerance=False)

    estimate_hist.append([x_hat, y_hat, theta_hat, v_hat, delta_hat])
    ukf_P_hist.append(ukf.P.copy())

    if prev_closest_idx is None:
        closest_idx = closest_point_idx(path_points, x_hat, y_hat)
    else:
        closest_idx = closest_point_idx_local(path_points, x_hat, y_hat, prev_closest_idx)
    assert closest_idx is not None, 'No closest point found'
    closest_idx_hist.append(closest_idx)
    prev_closest_idx = closest_idx

    lookahead_distance = get_lookahead_distance(v_hat)
    target_idx = get_lookahead_idx(path_points, closest_idx, lookahead_distance)

    target_point = path_points[target_idx]
    target_heading = path_headings[target_idx]
    target_curvature = path_curvatures[target_idx]
    target_dcurvature = path_dcurvatures[target_idx]
    target_velocity = velocity_profile[target_idx]

    # Pure pursuit controller
    dist_to_target = sqrt((target_point[0] - x_hat)**2 + (target_point[1] - y_hat)**2)
    angle_to_target = atan2(target_point[1] - y_hat, target_point[0] - x_hat) - theta_hat
    target_delta = atan2(2*model_params['l']*sin(angle_to_target), dist_to_target)

    # Compute the control inputs (with saturation)
    a = clamp(a_controller.step(target_velocity - v_hat), -max_linear_acceleration, max_linear_acceleration)
    delta_dot = clamp(delta_dot_controller.step(wrap_to_pi(target_delta - delta_hat)), -max_steering_rate, max_steering_rate)

    # Simulate the bicycle
    state = kinematic_bicycle_model(state, [a, delta_dot], model_params)
    u_hist.append([a, delta_dot])
t_hist = [i * model_params['dt'] for i in range(num_steps)]

# # Plot state_hist on top of path_points
# plt.figure()
# plt.plot([p[0] for p in path_points], [p[1] for p in path_points], '.', label='path')
# plt.plot([p[0] for p in state_hist], [p[1] for p in state_hist], '.', label='ego')
# plt.axis('equal')
# plt.title('BEV')
# plt.legend()
# plt.show()

# %%
# Plot simulation data
from filterpy.stats import plot_covariance

EGO_COLOR = 'tab:orange'
EGO_ESTIMATE_COLOR = 'tab:red'
EGO_ACTUATION_COLOR = 'tab:green'
TARGET_COLOR = 'tab:blue'
TITLE = "Simulation Data"
FIGSIZE_MULTIPLIER = 1.5
fig = plt.figure(figsize=(6.4 * FIGSIZE_MULTIPLIER, 4.8 * FIGSIZE_MULTIPLIER), constrained_layout=True)
suptitle = fig.suptitle(TITLE)
subfigs = fig.subfigures(2, 2)
COV_INTERVAL_S = 5 # number of seconds between covariance ellipses
uncertainty_std = 1 # number of standard deviations to plot for uncertainty

# BEV plot
ax_bev = subfigs[0][0].add_subplot(111)
ax_bev.plot([p[0] for p in path_points], [p[1] for p in path_points], '.', label='path', markersize=0.1, color=TARGET_COLOR)
ego_position = ax_bev.plot([p[0] for p in state_hist], [p[1] for p in state_hist], '.', markersize=0.1, color=EGO_COLOR, label='ego')[0]
ego_position_estimate = ax_bev.plot([p[0] for p in estimate_hist], [p[1] for p in estimate_hist], '.', markersize=0.1, color=EGO_ESTIMATE_COLOR, label='ego estimate')[0]
# Velocity plot
ax_velocity = subfigs[0][1].add_subplot(111)
ax_velocity.plot(t_hist, [velocity_profile[idx] for idx in closest_idx_hist], label=r"$v_d$", color=TARGET_COLOR) # target velocity
ax_velocity.plot(t_hist, [p[3] for p in state_hist], label=r"$v$", color=EGO_COLOR) # velocity
ax_velocity.plot(t_hist, [p[3] for p in estimate_hist], label=r"$\hat{v}$", color=EGO_ESTIMATE_COLOR) # velocity estimate
ax_velocity.fill_between(t_hist, [p[3] - uncertainty_std*np.sqrt(ukf_P[3,3]) for p, ukf_P in zip(estimate_hist, ukf_P_hist)], [p[3] + uncertainty_std*np.sqrt(ukf_P[3,3]) for p, ukf_P in zip(estimate_hist, ukf_P_hist)], alpha=0.2, color=EGO_ESTIMATE_COLOR)
# Heading&Steering plot
ax_heading_steering = subfigs[1][0].subplots(2, 1, sharex=True)
ax_heading = ax_heading_steering[0]
ax_heading.plot(t_hist, np.unwrap([path_headings[idx] for idx in closest_idx_hist]), label=r"$\theta_d$", color=TARGET_COLOR) # target heading
ax_heading.plot(t_hist, np.unwrap([p[2] for p in state_hist]), label=r"$\theta$", color=EGO_COLOR) # heading
ax_heading.plot(t_hist, np.unwrap([p[2] for p in estimate_hist]), label=r"$\hat{\theta}$", color=EGO_ESTIMATE_COLOR) # heading estimate
ax_heading.fill_between(t_hist, np.unwrap([p[2] - uncertainty_std*np.sqrt(ukf_P[2,2]) for p, ukf_P in zip(estimate_hist, ukf_P_hist)]), np.unwrap([p[2] + uncertainty_std*np.sqrt(ukf_P[2,2]) for p, ukf_P in zip(estimate_hist, ukf_P_hist)]), alpha=0.2, color=EGO_ESTIMATE_COLOR)
ax_steering = ax_heading_steering[1]
ax_steering.plot(t_hist, [p[4] for p in state_hist], label=r"$\delta$", color=EGO_COLOR) # steering
ax_steering.plot(t_hist, [p[4] for p in estimate_hist], label=r"$\hat{\delta}$", color=EGO_ESTIMATE_COLOR) # steering estimate
ax_steering.fill_between(t_hist, [p[4] - uncertainty_std*np.sqrt(ukf_P[4,4]) for p, ukf_P in zip(estimate_hist, ukf_P_hist)], [p[4] + uncertainty_std*np.sqrt(ukf_P[4,4]) for p, ukf_P in zip(estimate_hist, ukf_P_hist)], alpha=0.2, color=EGO_ESTIMATE_COLOR)
# Control signals plot
axes_control = subfigs[1][1].subplots(2, 1, sharex=True)
axes_control[0].plot(t_hist, [u[0] for u in u_hist], label=r"$a$", color=EGO_ACTUATION_COLOR) # a
axes_control[1].plot(t_hist, [u[1] for u in u_hist], label=r"$\dot{\delta}$", color=EGO_ACTUATION_COLOR) # delta_dot

# # Plot covariance ellipses
# plt.sca(ax_bev)
# for est, ukf_P in zip(estimate_hist[::int(COV_INTERVAL_S/model_params['dt'])], ukf_P_hist[::int(COV_INTERVAL_S/model_params['dt'])]):
#     plot_covariance(est[:2], std=100, cov=ukf_P[0:2, 0:2], facecolor='none', edgecolor=EGO_ESTIMATE_COLOR)

# Finalize plots
ax_bev.axis('equal')
ax_bev.set_title('BEV')
ax_bev.legend(markerscale=50)
ax_velocity.set_title("Velocity over time")
ax_velocity.set_xlabel(r'Time ($s$)')
ax_velocity.set_ylabel(r'Velocity ($m/s$)')
ax_velocity.legend()
ax_heading.set_title("Heading and Steering over time")
ax_heading.set_ylabel(r'Heading ($rad$)')
ax_heading.legend()
ax_steering.set_xlabel(r'Time ($s$)')
ax_steering.set_ylabel(r'Steering ($rad$)')
ax_steering.legend()
axes_control[0].set_title("Control signals over time")
axes_control[0].set_ylabel(r'Acceleration ($m/s^2$)')
axes_control[0].legend()
axes_control[1].set_xlabel(r'Time ($s$)')
axes_control[1].set_ylabel(r'Steering rate ($rad/s$)')
axes_control[1].legend()

#%%
# Run the solver for the last N outputs

# Note that the indexing here is slightly different from the one in the loop, because in the loop, we are getting data before we append elements to u_hist, estimate_hist, and closest_idx_hist
res = estimate_state(None, [o + np.array([0,0,0,0,0,0]) for o in output_hist[-N:]], u_hist[-N:-1], estimate_hist[-N:-1], closest_idx_hist[-N:-1], path_points, path_headings, velocity_profile, [C]*N, dt=model_params['dt'], l=model_params['l'], N=N, enable_fault_tolerance=True)

# solver_setup = get_solver_setup([o + np.array([0,0,0,0,0,0]) for o in output_hist[-N:]], u_hist[-N:-1], closest_idx_hist[-N:-1], path_points, path_headings, velocity_profile, [C]*N, dt=model_params['dt'], l=model_params['l'])
# print(solver_setup['Phi'] @ solver_setup['output_hist_no_input_effects'][0][:5] - solver_setup['Y'])

# %%
# Retroactively calculate from data the modelling error and the noise
errors = []

# Arbitrarily ignore the first 500 data points because they are not at steady state
for i in range(max(N, 500), len(output_hist)+1):
    solver_setup = get_solver_setup(output_hist[i-N:i], u_hist[i-N:i-1], closest_idx_hist[i-N:i-1], path_points, path_headings, velocity_profile, [C]*N, dt=model_params['dt'], l=model_params['l'])
    error = solver_setup['Phi'] @ solver_setup['output_hist_no_input_effects'][0][:5] - solver_setup['Y']
    error[:,2] = wrap_to_pi(error[:,2])
    error[:,4] = wrap_to_pi(error[:,4])
    error[:,5] = wrap_to_pi(error[:,5])
    errors.append(error)

errors_tensor = np.array(errors)

max_errors_over_time = abs(errors_tensor).max(axis=1)
x = np.arange(len(max_errors_over_time))
for i in range(6):
    plt.plot(x, max_errors_over_time[:,i], ".-", label=f"sensor {i}")
plt.xlabel("time step")
plt.ylabel("max error")
plt.title("Max error per sensor over time")
plt.legend()
plt.show()

print("Max error per sensor and per time step in an interval")
print(abs(errors_tensor).max(axis=0))

print("95th percentile error per sensor per time step in an interval")
print(np.percentile(abs(errors_tensor), 95, axis=0))

print("90th percentile error per sensor per time step in an interval")
print(np.percentile(abs(errors_tensor), 90, axis=0))

# %%

# plt.show()

#%%
# Generate a GIF
def generate_gif():
    from matplotlib.animation import PillowWriter, FuncAnimation
    ANIM_TITLE_FORMAT = 'Simulation Playback (Current time: ${:.2f}$ s)'

    # add time cursors
    time_cursors = []
    time_cursors.append(ax_velocity.axvline(0, color='k'))
    time_cursors.append(ax_heading.axvline(0, color='k'))
    time_cursors.append(axes_control[0].axvline(0, color='k'))
    time_cursors.append(axes_control[1].axvline(0, color='k'))
    def animate(i):
        # ================= Update title =================
        suptitle.set_text(ANIM_TITLE_FORMAT.format(i * model_params['dt']))

        # =============== Plot ego position ===============
        # For a moving position, use this:
        #ego_position.set_data(state_hist[i][0], state_hist[i][1])
        # To plot the entire trajectory, use this:
        ego_position.set_data([p[0] for p in state_hist[:i+1]], [p[1] for p in state_hist[:i+1]])
        ego_position_estimate.set_data([p[0] for p in estimate_hist[:i+1]], [p[1] for p in estimate_hist[:i+1]])

        # =============== Plot time cursors ===============
        for time_cursor in time_cursors:
            time_cursor.set_xdata([i * model_params['dt']])

        return suptitle, ego_position, *time_cursors

    anim_interval_ms = 2000
    anim = FuncAnimation(fig, animate, frames=range(0, num_steps, int(anim_interval_ms / 1000 / model_params['dt'])), interval=anim_interval_ms)

    start = time.perf_counter()
    # anim.save('zero_state.gif', writer=PillowWriter(fps=1000/anim_interval_ms))

    from IPython.core.display import HTML
    html = HTML(anim.to_jshtml())
    print(f"Animation generation took {time.perf_counter() - start:.2f} seconds")
    plt.close(fig)

    return html

# generate_gif()

# #%%

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
