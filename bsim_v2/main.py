#%%

import json
import math
import matplotlib.pyplot as plt
import numpy as np
import time
from math import pi, sin, cos, atan2, sqrt
from typing import List, Tuple
plt.rcParams['text.usetex'] = True

#%%

model_params = {
    'dt': 0.01,
    'l': 0.5,
}

# Kinematic bicycle model
def kinematic_bicycle_model(state, input, params):
    # state = [x, y, theta, v, delta]
    # input = [a, delta_dot]
    # params = {dt, l}
    dt = params['dt']
    l = params['l']

    x = state[0]
    y = state[1]
    theta = state[2]
    v = state[3]
    delta = state[4]

    a = input[0]
    delta_dot = input[1]

    x_dot = v * np.cos(theta)
    y_dot = v * np.sin(theta)
    theta_dot = v / l * np.tan(delta)

    x += x_dot * dt
    y += y_dot * dt
    theta += theta_dot * dt
    v += a * dt
    delta += delta_dot * dt

    return np.array([x, y, theta, v, delta])

# Derived from research-jackal
def generate_circle_approximation(center, radius, num_points):
    """
    Generates a list of points on a circle.
    Arguments:
        center: The center of the circle.
        radius: The radius of the circle.
        num_points: The number of points to generate. The more points, the more accurate the approximation.
    Returns:
        points: A list of points on the figure eight shape.
        headings: A list of headings (θ) at each point.
        curvatures: A list of curvatures (κ) at each point.
        dK_ds_list: A list of dκ_ds at each point. Where κ is the curvature and s is the arc length.
    
    Derivation: https://github.com/ben-z/research-sensor-attack/blob/e20c7b02cf6aca6c18c37976550c03606919192a/curves.py#L173-L191
    """
    a = radius

    points = []
    headings = []
    curvatures = []
    dK_ds_list = []
    for i in range(num_points):
        t = 2 * pi * i / num_points
        points.append([
            center[0] + a * cos(t),
            center[1] + a * sin(t)
        ])
        headings.append(atan2(a*cos(t), -a*sin(t)))
        # This could be simplified to 1/a, but we leave it as is for consistency with the derivation.
        curvatures.append(1/sqrt(a**2*sin(t)**2 + a**2*cos(t)**2))
        dK_ds_list.append(0) # circles have constant curvature
    return points, headings, curvatures, dK_ds_list


# Derived from research-jackal
def generate_figure_eight_approximation(center, length, width, num_points):
    """
    Generates a list of points on a figure eight shape.
    Arguments:
        center: The center of the figure eight shape.
        length: The length of the figure eight shape.
        width: The width of the figure eight shape.
        num_points: The number of points to generate. The more points, the more accurate the approximation.
    Returns:
        points: A list of points on the figure eight shape.
        headings: A list of headings (θ) at each point.
        curvatures: A list of curvatures (κ) at each point.
        dK_ds_list: A list of dκ_ds at each point. Where κ is the curvature and s is the arc length.

    The formula used to generate the points is:
        x = a * sin(t)
        y = b * sin(2t)/2
    where a = length / 2 and b = width.

    Supplementary visualization:
    https://www.desmos.com/calculator/fciqxay3p2
    Derivation:
    https://github.com/ben-z/research-sensor-attack/blob/e20c7b02cf6aca6c18c37976550c03606919192a/curves.py#L153-L171
    """
    a = length / 2
    b = width

    points = []
    headings = []
    curvatures = []
    dK_ds_list = []
    for i in range(num_points):
        # t is an arbitrary parameter that is used to generate the points
        # The result is known as an arbitrary-speed curve
        t = 2 * pi * i / num_points
        x = center[0] + a * sin(t)
        y = center[1] + b * (sin(t * 2) / 2)
        points.append([x, y])
        headings.append(atan2(b * cos(t * 2), a * cos(t)))
        curvatures.append((a*b*sin(t)*cos(2*t) - 2*a*b*sin(2*t)*cos(t))/(a**2*cos(t)**2 + b**2*cos(2*t)**2)**(3/2))
        dK_ds_list.append((-3*a*b*cos(t)*cos(2*t)/(a**2*cos(t)**2 + b**2*cos(2*t)**2)**(3/2) + (3*a**2*sin(t)*cos(t) + 6*b**2*sin(2*t)*cos(2*t))*(a*b*sin(t)*cos(2*t) - 2*a*b*sin(2*t)*cos(t))/(a**2*cos(t)**2 + b**2*cos(2*t)**2)**(5/2))/sqrt(a**2*cos(t)**2 + b**2*cos(2*t)**2))
    return points, headings, curvatures, dK_ds_list


class PIDController:
    def __init__(self, kp, ki, kd, dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt

        self.integral = 0
        self.prev_error = 0

    def step(self, error):
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error

        return self.kp * error + self.ki * self.integral + self.kd * derivative

def closest_point_idx(points, x, y):
    closest_idx = None
    closest_dist = None
    for i, p in enumerate(points):
        dist = sqrt((p[0] - x)**2 + (p[1] - y)**2)
        if closest_dist is None or dist < closest_dist:
            closest_dist = dist
            closest_idx = i
    return closest_idx

def closest_point_idx_local(points, x, y, prev_idx):
    """
    Performs a local search for the closest point.
    """
    closest_idx_forward = None
    closest_dist_forward = None
    # search forwards
    for i in range(prev_idx, prev_idx + len(points)):
        idx = i % len(points)
        dist = sqrt((points[idx][0] - x)**2 + (points[idx][1] - y)**2)
        if closest_dist_forward is None or dist < closest_dist_forward:
            closest_dist_forward = dist
            closest_idx_forward = idx
        else:
            break

    closest_idx_backward = None
    closest_dist_backward = None
    # search backwards
    for i in range(prev_idx, prev_idx - len(points), -1):
        idx = i % len(points)
        dist = sqrt((points[idx][0] - x)**2 + (points[idx][1] - y)**2)
        if closest_dist_backward is None or dist < closest_dist_backward:
            closest_dist_backward = dist
            closest_idx_backward = idx
        else:
            break
    
    assert closest_dist_forward is not None, 'No closest point found in the forward direction'
    assert closest_dist_backward is not None, 'No closest point found in the backward direction'
    if closest_dist_forward < closest_dist_backward:
        return closest_idx_forward
    else:
        return closest_idx_backward


def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def walk_path(path_points, starting_idx, dist):
    remaining_dist = dist
    idx = starting_idx
    while remaining_dist > 0:
        if idx == len(path_points) - 1:
            idx = 0
        else:
            idx += 1
        remaining_dist -= sqrt((path_points[idx][0] - path_points[idx-1][0])**2 + (path_points[idx][1] - path_points[idx-1][1])**2)
    return idx

def clamp(x, lower, upper):
    return np.maximum(lower, np.minimum(x, upper))

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
path_points, path_headings, path_curvatures, path_dcurvatures = generate_figure_eight_approximation([0, 0], 2000, 1000, 1000)

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

velocity_profile_raw = np.sqrt(side_friction_factor * g / np.abs(path_curvatures))
velocity_profile = np.clip(velocity_profile_raw, min_linear_velocity, max_linear_velocity)
# velocity_profile = smooth_velocity_profile(velocity_profile, path_lengths, max_linear_acceleration)
velocity_profile = np.clip(velocity_profile, min_linear_velocity, max_linear_velocity)

# Plot the original velocity profile and the smoothed velocity profile
plt.figure()
plt.plot(velocity_profile, label='smoothed')
plt.plot(velocity_profile_raw, label='original')
plt.title('Velocity profile')
plt.ylim([0, 1.1 * max(velocity_profile)])
plt.legend()
plt.show()

# Plot the acceleration profile
plt.figure()


# Control the bicycle to follow the path
simulation_seconds = 30
num_steps = int(simulation_seconds / model_params['dt'])
x0 = np.array([200,0, pi/4, 1, 0])
state = x0
state_hist = []
estimate_hist = []
u_hist = []
closest_idx_hist = []
a_controller = PIDController(2, 0, 0, model_params['dt'])
delta_dot_controller = PIDController(5, 0, 0, model_params['dt'])
prev_closest_idx = None
for i in range(num_steps):
    state_hist.append(state)

    output = np.concatenate((state, [state[4]]))
    # attack
    if i * model_params['dt'] > 5:
        output[3] += 5

    # estimates
    x_hat = output[0]
    y_hat = output[1]
    theta_hat = output[2]
    v_hat = output[3]
    delta_hat = np.mean([output[4], output[5]])

    estimate_hist.append([x_hat, y_hat, theta_hat, v_hat, delta_hat])

    if prev_closest_idx is None:
        closest_idx = closest_point_idx(path_points, x_hat, y_hat)
    else:
        closest_idx = closest_point_idx_local(path_points, x_hat, y_hat, prev_closest_idx)
    assert closest_idx is not None, 'No closest point found'
    closest_idx_hist.append(closest_idx)
    prev_closest_idx = closest_idx

    lookahead_distance = get_lookahead_distance(v_hat)
    target_idx = walk_path(path_points, closest_idx, lookahead_distance)

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

# Plot state_hist on top of path_points
plt.figure()
plt.plot([p[0] for p in path_points], [p[1] for p in path_points], '.', label='path')
plt.plot([p[0] for p in state_hist], [p[1] for p in state_hist], '.', label='ego')
plt.axis('equal')
plt.title('BEV')
plt.legend()
plt.show()

# Plot simulation data
EGO_COLOR = 'tab:orange'
EGO_ESTIMATE_COLOR = 'tab:red'
EGO_ACTUATION_COLOR = 'tab:green'
TARGET_COLOR = 'tab:blue'
TITLE = "Simulation Data"
FIGSIZE_MULTIPLIER = 1.5
fig = plt.figure(figsize=[6.4 * FIGSIZE_MULTIPLIER, 4.8 * FIGSIZE_MULTIPLIER], constrained_layout=True)
suptitle = fig.suptitle(TITLE)
subfigs = fig.subfigures(2, 2)
# BEV plot
ax_bev = subfigs[0][0].add_subplot(111)
ax_bev.plot([p[0] for p in path_points], [p[1] for p in path_points], '.', label='path')
ego_position = ax_bev.plot([p[0] for p in state_hist], [p[1] for p in state_hist], '.', color=EGO_COLOR, label='ego')[0]
ego_position_estimate = ax_bev.plot([p[0] for p in estimate_hist], [p[1] for p in estimate_hist], '.', color=EGO_ESTIMATE_COLOR, label='ego estimate')[0]
# Velocity plot
ax_velocity = subfigs[0][1].add_subplot(111)
ax_velocity.plot(t_hist, [velocity_profile[idx] for idx in closest_idx_hist], label=r"$v_d$", color=TARGET_COLOR) # target velocity
ax_velocity.plot(t_hist, [p[3] for p in state_hist], label=r"$v$", color=EGO_COLOR) # velocity
ax_velocity.plot(t_hist, [p[3] for p in estimate_hist], label=r"$\hat{v}$", color=EGO_ESTIMATE_COLOR) # velocity estimate
# Heading plot
ax_heading = subfigs[1][0].add_subplot(111)
ax_heading.plot(t_hist, [path_headings[idx] for idx in closest_idx_hist], label=r"$\theta_d$", color=TARGET_COLOR) # target heading
ax_heading.plot(t_hist, [wrap_to_pi(p[2]) for p in state_hist], label=r"$\theta$", color=EGO_COLOR) # heading
ax_heading.plot(t_hist, [wrap_to_pi(p[2]) for p in estimate_hist], label=r"$\hat{\theta}$", color=EGO_ESTIMATE_COLOR) # heading estimate
# Control signals plot
axes_control = subfigs[1][1].subplots(2, 1, sharex=True)
axes_control[0].plot(t_hist, [u[0] for u in u_hist], label=r"$a$", color=EGO_ACTUATION_COLOR) # a
axes_control[1].plot(t_hist, [u[1] for u in u_hist], label=r"$\dot{\delta}$", color=EGO_ACTUATION_COLOR) # delta_dot

# Finalize plots
ax_bev.axis('equal')
ax_bev.set_title('BEV')
ax_bev.legend()
ax_velocity.set_title("Velocity over time")
ax_velocity.set_xlabel(r'Time ($s$)')
ax_velocity.set_ylabel(r'Velocity ($m/s$)')
ax_velocity.legend()
ax_heading.set_title("Heading over time")
ax_heading.set_xlabel(r'Time ($s$)')
ax_heading.set_ylabel(r'Heading ($rad$)')
ax_heading.legend()
axes_control[0].set_title("Control signals over time")
axes_control[0].set_ylabel(r'Acceleration ($m/s^2$)')
axes_control[0].legend()
axes_control[1].set_xlabel(r'Time ($s$)')
axes_control[1].set_ylabel(r'Steering rate ($rad/s$)')
axes_control[1].legend()
fig

#%%
# Generate a GIF
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
html

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
