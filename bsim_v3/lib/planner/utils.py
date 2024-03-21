from math import atan2, cos, pi, sin, sqrt

import numpy as np


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

    points = np.zeros((num_points, 2))
    headings = np.zeros(num_points)
    curvatures = np.zeros(num_points)
    dK_ds_list = np.zeros(num_points)

    for i in range(num_points):
        # t is an arbitrary parameter that is used to generate the points
        # The result is known as an arbitrary-speed curve
        t = 2 * pi * i / num_points
        x = center[0] + a * sin(t)
        y = center[1] + b * (sin(t * 2) / 2)
        points[i] = [x, y]
        headings[i] = atan2(b * cos(t * 2), a * cos(t))
        curvatures[i] = (
            (a * b * sin(t) * cos(2 * t) - 2 * a * b * sin(2 * t) * cos(t))
            / (a**2 * cos(t) ** 2 + b**2 * cos(2 * t) ** 2) ** (3 / 2)
        )
        dK_ds_list[i] = (
            (
                -3
                * a
                * b
                * cos(t)
                * cos(2 * t)
                / (a**2 * cos(t) ** 2 + b**2 * cos(2 * t) ** 2) ** (3 / 2)
                + (3 * a**2 * sin(t) * cos(t) + 6 * b**2 * sin(2 * t) * cos(2 * t))
                * (a * b * sin(t) * cos(2 * t) - 2 * a * b * sin(2 * t) * cos(t))
                / (a**2 * cos(t) ** 2 + b**2 * cos(2 * t) ** 2) ** (5 / 2)
            )
            / sqrt(a**2 * cos(t) ** 2 + b**2 * cos(2 * t) ** 2)
        )
    return points, headings, curvatures, dK_ds_list


def closest_point_idx(points, x, y):
    closest_idx = None
    closest_dist = None
    for i, p in enumerate(points):
        dist = sqrt((p[0] - x) ** 2 + (p[1] - y) ** 2)
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
        dist = sqrt((points[idx][0] - x) ** 2 + (points[idx][1] - y) ** 2)
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
        dist = sqrt((points[idx][0] - x) ** 2 + (points[idx][1] - y) ** 2)
        if closest_dist_backward is None or dist < closest_dist_backward:
            closest_dist_backward = dist
            closest_idx_backward = idx
        else:
            break

    assert (
        closest_dist_forward is not None
    ), "No closest point found in the forward direction"
    assert (
        closest_dist_backward is not None
    ), "No closest point found in the backward direction"
    if closest_dist_forward < closest_dist_backward:
        return closest_idx_forward
    else:
        return closest_idx_backward


def get_lookahead_idx(path_points, starting_idx, dist):
    remaining_dist = dist
    idx = starting_idx
    while remaining_dist > 0:
        if idx == len(path_points) - 1:
            idx = 0
        else:
            idx += 1
        remaining_dist -= sqrt(
            (path_points[idx][0] - path_points[idx - 1][0]) ** 2
            + (path_points[idx][1] - path_points[idx - 1][1]) ** 2
        )
    return idx


def calc_target_velocity(curvatures, max_speed, side_friction_factor_base=0.1):
    max_curvature = max(np.abs(curvatures))
    min_turning_radius = 1 / max_curvature

    # unitless. Determines the maximum lateral force a car can take.
    # 0.15 for urban,
    # 0.1 for highway,
    # (much) higher for racecars.
    side_friction_factor = side_friction_factor_base + min_turning_radius / 100

    velocities = np.clip(
        np.sqrt(side_friction_factor * 9.81 / np.clip(np.abs(curvatures), 1e-6, None)),
        0,
        max_speed,
    )

    return velocities