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
# Set up the environment

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

# %%
# Add imports and set up parameters for the simulation

import cvxpy as cp
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import time
from math import pi, sin, cos, atan2, sqrt
from multiprocessing import Pool
from numpy.typing import NDArray
from typing import List, Tuple, Any, Optional
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
    MAX_POOL_SIZE,
)
from fault_generators import (
    sensor_bias_fault,
    intermittent_fault,
    drift_fault,
)

plt.rcParams["text.usetex"] = True
np.set_printoptions(suppress=True)

g = 9.81  # m/s^2

model_params = {
    "dt": 0.01,
    "l": 0.5,
    "max_linear_acceleration": 4 * g,  # m/s^2
    "max_steering_rate": 10,  # rad/s
    "max_linear_velocity": 300 / 3.6,  # m/s
    "min_linear_velocity": 0.1,  # m/s
}

(
    path_points,
    path_headings,
    path_curvatures,
    path_dcurvatures,
) = generate_figure_eight_approximation([0, 0], 2000, 1000, 100000)

path_lengths = calculate_segment_lengths(path_points)

path_max_curvature = max(np.abs(path_curvatures))
path_min_turning_radius = 1 / path_max_curvature
print(
    f"Maximum curvature: {path_max_curvature:.4f} m^-1"
    f" (minimum turning radius: {path_min_turning_radius:.2f} m)"
)

# unitless. Determines the maximum lateral force a car can take.
# Typical values are 0.15 for urban, 0.1 for highway, (much) higher for racecars.
side_friction_factor = 0.1 + path_min_turning_radius / 100

velocity_profile_raw = np.sqrt(
    side_friction_factor * g / np.clip(np.abs(path_curvatures), 1e-6, None)
)
velocity_profile = np.clip(
    velocity_profile_raw,
    model_params["min_linear_velocity"],
    model_params["max_linear_velocity"],
)

# # Plot the original velocity profile and the processed velocity profile
# plt.figure()
# plt.plot(velocity_profile, label='processed')
# plt.plot(velocity_profile_raw, label='original')
# plt.title('Velocity profile')
# plt.ylim([0, 1.1 * max(velocity_profile)])
# plt.legend()
# plt.show()

# Set up the system
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

# %%

# Set up CLI
import typer

app = typer.Typer()

# %%


@app.command()
def run(
    output: str,
    exp_name: Optional[str] = None,
    dry_run: bool = True,
    num_passes: int = 1,
    overwrite: bool = False,
    fault_selection: str = "all",
    sensor_selection: str = "all",
):  # Batch experiments
    fault_specs = []
    file_name = output
    # defaults to the name of the file
    exp_name = exp_name or os.path.basename(file_name).split(".")[0]

    print(f"Experiment name: {exp_name}. Saving to {file_name}")

    if os.path.exists(file_name):
        if overwrite:
            print(f"File {file_name} already exists. Overwriting.")
            os.remove(file_name)
        else:
            raise ValueError(f"File {file_name} already exists")

    supported_faults = set(["all", "bias", "spike", "noise", "drift"])
    selected_faults = set(fault_selection.split(","))
    if not supported_faults.issuperset(selected_faults):
        raise ValueError(f"Fault selection must be a subset of {supported_faults}. The following are not supported: {selected_faults - supported_faults}")
    if "all" in selected_faults:
        selected_faults = supported_faults

    supported_sensors = set(range(2,6))
    if sensor_selection == "all":
        selected_sensors = supported_sensors
    else:
        selected_sensors = set(map(int, sensor_selection.split(",")))
        if not supported_sensors.issuperset(selected_sensors):
            raise ValueError(f"Sensor selection must be a subset of {supported_sensors}. The following are not supported: {selected_sensors - supported_sensors}")

    # Inject fault at different points of the simulation
    for start_t in [10, 15, 20, 25, 30, 35, 40]:
        if "bias" in selected_faults:
            #############################################
            # Bias Faults
            #############################################

            # heading sensor
            if 2 in selected_sensors:
                for bias in np.arange(-np.pi/4, np.pi/4 + sys.float_info.epsilon, 0.005):
                    fault_specs.append(
                        {
                            "fn": "sensor_bias_fault",
                            "kwargs": {"start_t": start_t, "sensor_idx": 2, "bias": bias},
                        }
                    )
            # velocity sensor
            if 3 in selected_sensors:
                for bias in np.arange(-5, 5 + sys.float_info.epsilon, 0.5):
                    fault_specs.append(
                        {
                            "fn": "sensor_bias_fault",
                            "kwargs": {"start_t": start_t, "sensor_idx": 3, "bias": bias},
                        }
                    )

            # steering angle sensor 1
            if 4 in selected_sensors:
                for bias in np.arange(-np.pi/8, np.pi/8 + sys.float_info.epsilon, 0.005):
                    fault_specs.append(
                        {
                            "fn": "sensor_bias_fault",
                            "kwargs": {"start_t": start_t, "sensor_idx": 4, "bias": bias},
                        }
                    )

            # steering angle sensor 2
            if 5 in selected_sensors:
                for bias in np.arange(-np.pi/8, np.pi/8 + sys.float_info.epsilon, 0.005):
                    fault_specs.append(
                        {
                            "fn": "sensor_bias_fault",
                            "kwargs": {"start_t": start_t, "sensor_idx": 5, "bias": bias},
                        }
                    )

        if "spike" in selected_faults:
            #############################################
            # Spike Faults
            #############################################

            # heading sensor
            if 2 in selected_sensors:
                for spike_value in np.arange(-np.pi/4, np.pi/4 + sys.float_info.epsilon, 0.1):
                    for duration in np.arange(0.1, 2 + sys.float_info.epsilon, 0.1):
                        fault_specs.append(
                            {
                                "fn": "spike_fault",
                                "kwargs": {
                                    "start_t": start_t,
                                    "sensor_idx": 2,
                                    "spike_value": spike_value,
                                    "duration": duration,
                                },
                            }
                        )

            # velocity sensor
            if 3 in selected_sensors:
                for spike_value in np.arange(-5, 5 + sys.float_info.epsilon, 0.5):
                    for duration in np.arange(0.1, 2 + sys.float_info.epsilon, 0.1):
                        fault_specs.append(
                            {
                                "fn": "spike_fault",
                                "kwargs": {
                                    "start_t": start_t,
                                    "sensor_idx": 3,
                                    "spike_value": spike_value,
                                    "duration": duration,
                                },
                            }
                    )

            # steering angle sensor 1
            if 4 in selected_sensors:
                for spike_value in np.arange(-np.pi/8, np.pi/8 + sys.float_info.epsilon, 0.1):
                    for duration in np.arange(0.1, 2 + sys.float_info.epsilon, 0.1):
                        fault_specs.append(
                            {
                                "fn": "spike_fault",
                                "kwargs": {
                                    "start_t": start_t,
                                    "sensor_idx": 4,
                                    "spike_value": spike_value,
                                    "duration": duration,
                                },
                            }
                        )

            # steering angle sensor 2
            if 5 in selected_sensors:
                for spike_value in np.arange(-np.pi/8, np.pi/8 + sys.float_info.epsilon, 0.1):
                    for duration in np.arange(0.1, 2 + sys.float_info.epsilon, 0.1):
                        fault_specs.append(
                            {
                                "fn": "spike_fault",
                                "kwargs": {
                                    "start_t": start_t,
                                    "sensor_idx": 5,
                                    "spike_value": spike_value,
                                    "duration": duration,
                                },
                            }
                        )

        if "noise" in selected_faults:
            #############################################
            # Noise Faults
            #############################################

            # heading sensor
            if 2 in selected_sensors:
                for noise_level in np.arange(0, np.pi/4 + sys.float_info.epsilon, 0.01):
                    fault_specs.append(
                        {
                            "fn": "random_noise_fault",
                            "kwargs": {
                                "start_t": start_t,
                                "sensor_idx": 2,
                                "noise_level": noise_level,
                            },
                        }
                    )

            # velocity sensor
            if 3 in selected_sensors:
                for noise_level in np.arange(0, 5 + sys.float_info.epsilon, 0.5):
                    fault_specs.append(
                        {
                            "fn": "random_noise_fault",
                            "kwargs": {
                                "start_t": start_t,
                                "sensor_idx": 3,
                                "noise_level": noise_level,
                            },
                        }
                    )

            # steering angle sensor 1
            if 4 in selected_sensors:
                for noise_level in np.arange(0, np.pi/8 + sys.float_info.epsilon, 0.01):
                    fault_specs.append(
                        {
                            "fn": "random_noise_fault",
                            "kwargs": {
                                "start_t": start_t,
                                "sensor_idx": 4,
                                "noise_level": noise_level,
                            },
                        }
                    )

            # steering angle sensor 2
            if 5 in selected_sensors:
                for noise_level in np.arange(0, np.pi / 4 + sys.float_info.epsilon, 0.01):
                    fault_specs.append(
                        {
                            "fn": "random_noise_fault",
                            "kwargs": {
                                "start_t": start_t,
                                "sensor_idx": 5,
                                "noise_level": noise_level,
                            },
                        }
                    )

        if "drift" in selected_faults:
            #############################################
            # Drift Faults
            #############################################

            # heading sensor
            if 2 in selected_sensors:
                for drift_rate in np.arange(-np.pi/2, np.pi/2 + sys.float_info.epsilon, 0.005):
                    fault_specs.append(
                        {
                            "fn": "drift_fault",
                            "kwargs": {
                                "start_t": start_t,
                                "sensor_idx": 2,
                                "drift_rate": drift_rate,
                            },
                        }
                    )

            # velocity sensor
            if 3 in selected_sensors:
                for drift_rate in np.arange(-15, 15 + sys.float_info.epsilon, 0.5):
                    fault_specs.append(
                        {
                            "fn": "drift_fault",
                            "kwargs": {
                                "start_t": start_t,
                                "sensor_idx": 3,
                                "drift_rate": drift_rate,
                            },
                        }
                    )

            # steering angle sensor 1
            if 4 in selected_sensors:
                for drift_rate in np.arange(-np.pi / 16, np.pi / 16 + sys.float_info.epsilon, 0.005):
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

            # steering angle sensor 2
            if 5 in selected_sensors:
                for drift_rate in np.arange(-np.pi / 16, np.pi / 16 + sys.float_info.epsilon, 0.005):
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

    print(
        f"Running {len(fault_specs)} experiments for {num_passes} pass(es). Total: {len(fault_specs) * num_passes} experiments"
    )
    print(f"{MAX_POOL_SIZE=}")

    simulation_seconds = 50
    num_steps = int(simulation_seconds / model_params["dt"])

    if dry_run:
        print("Dry run. Not running experiments.")
        return

    for i in range(1, num_passes + 1):
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
        print(
            f"Experiment pass {i}/{num_passes} took {time.perf_counter() - start:.2f} seconds"
        )


if __name__ == "__main__":
    app()
