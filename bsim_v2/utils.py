import numpy as np
from math import pi, sin, cos, atan2, sqrt

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

def get_lookahead_idx(path_points, starting_idx, dist):
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

def walk_trajectory_by_durations(path_points, velocities, starting_idx, durations):
    assert all(d >= 0 for d in durations), "durations must be non-negative"
    assert len(durations) > 0, "Must have a duration"

    idx = starting_idx - 1
    remaining_segment_duration = 0.0
    indices = []
    for duration in durations:
        # first travel the untravelled portion of the current segment
        remaining_duration = max(0.0, duration - remaining_segment_duration)
        remaining_segment_duration -= duration - remaining_duration

        # at least one of two scenarios are possible here:
        # - we have used up the remaining segment duration, thus go into the loop to go onto the next segment
        # - we have not used up te remaining segment duration, so we stay on the current segment.
        # If both are true, we arbitrarily choose to record the existing segment index instead of the new one.
        # This is okay because we are already doing floating point calculations (minor errors are tolerable).
        while remaining_duration > 1e-15:
            # if this assertion is violated, then we have travelled more than the remaining duration, which is not possible
            assert -1e-15 < remaining_segment_duration < 1e-15, "remaining segment duration must be zero in the beginning of this loop"
            if remaining_segment_duration < 1e-15:
                # we have used up the remaining segment duration, progress onto the next segment
                idx = (idx+1)%len(path_points)

            segment_dist = sqrt((path_points[(idx+1)%len(path_points)][0] - path_points[idx][0])**2 + (path_points[(idx+1)%len(path_points)][1] - path_points[idx][1])**2)
            segment_duration = segment_dist / velocities[idx]
            
            remaining_segment_duration = max(0, segment_duration - remaining_duration)
            remaining_duration -= segment_duration - remaining_segment_duration

        indices.append(idx)
    return indices

