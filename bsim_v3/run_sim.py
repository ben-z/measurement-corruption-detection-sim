import glob
import json
import os
import random
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import typer
from more_itertools import powerset
# ----- Local imports (your existing modules) -----
from lib.controllers.pure_pursuit import \
    KinematicBicycle5StatePurePursuitController
from lib.detectors.detector import CalcValidityMetadata, LookAheadDetector
from lib.estimators.simple_ukf import SimpleUKF
from lib.fault_generators import (drift_fault, random_noise_fault,
                                  sensor_bias_fault, spike_fault)
from lib.planners.static import StaticFigureEightPlanner
from lib.planners.utils import calc_target_velocity
from lib.plants.kinematic_bicycle import \
    KinematicBicycle5StateRearWheelRefPlant
from lib.sensors.kinematic_bicycle_race_day import \
    KinematicBicycleRaceDaySensor
from tqdm import tqdm

# ----- CLI setup with Typer -----
app = typer.Typer(
    help="Run multiple simulations with a chosen fault type and save to Parquet."
)

# ----- Disable multithreading for numpy (optional) -----
NUM_THREADS = "1"
os.environ["OMP_NUM_THREADS"] = NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS
os.environ["MKL_NUM_THREADS"] = NUM_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS


# -----------------------------------------------------------------------------
# Ranges for random fault parameter generation
# -----------------------------------------------------------------------------
FAULT_RANGES = {
    "bias": {
        "heading": (-np.pi, np.pi),
        "velocity": (-10, 10),
        "steering": (-np.pi / 2, np.pi / 2),
    },
    "spike": {
        "heading": (-np.pi, np.pi),
        "velocity": (-10, 10),
        "steering": (-np.pi / 2, np.pi / 2),
    },
    "noise": {
        "heading": (0, np.pi),  # amplitude
        "velocity": (0, 10),  # amplitude
        "steering": (0, np.pi / 2),  # amplitude
    },
    "drift": {
        "heading": (-np.pi, np.pi),  # rad/s
        "velocity": (-10, 10),  # m/s^2
        "steering": (-np.pi / 2, np.pi / 2),  # rad/s
    },
}
FAULT_TYPES = list(FAULT_RANGES.keys())

# Direct mapping from sensor_idx to the type in FAULT_RANGES
PARAM_RANGE_KEY = {
    0: None,  # x-position
    1: None,  # y-position
    2: "heading",
    3: "velocity",
    4: "velocity",
    5: "steering",
}

flatten = lambda t: [item for sublist in t for item in sublist]

def expand_glob(
    ctx: typer.Context, param: typer.CallbackParam, values: List[Path]
) -> List[Path]:
    expanded_files = []
    for value in values:
        if (
            "*" in value.as_posix()
            or "?" in value.as_posix()
            or "[" in value.as_posix()
        ):
            expanded_files.extend(Path().glob(value.as_posix()))
        else:
            expanded_files.append(value.resolve())

    return expanded_files


def create_fault_fn(fault_type, sensor_idx):
    """
    Given a fault type (bias, spike, noise, drift) and a sensor index,
    return a function that, given tstart, produces the final fault function.
    Also returns the fault parameters.
    """
    # Determine the parameter-range key from the dictionary
    fkey = PARAM_RANGE_KEY.get(sensor_idx)
    # If None, fallback to "steering" or skip entirely.
    if fkey is None:
        raise ValueError(f"Sensor {sensor_idx} does not have a fault parameter.")

    low, high = FAULT_RANGES[fault_type][fkey]
    param = random.uniform(low, high)

    if fault_type == "bias":
        return lambda tstart: sensor_bias_fault(tstart, sensor_idx, bias=param), {"bias": param}
    elif fault_type == "spike":
        duration = random.randint(1, 10)
        return lambda tstart: spike_fault(
            tstart, sensor_idx, amplitude=param, duration=duration
        ), {"amplitude": param, "duration": duration}
    elif fault_type == "noise":
        return lambda tstart: random_noise_fault(tstart, sensor_idx, amplitude=param), {"amplitude": param}
    elif fault_type == "drift":
        return lambda tstart: drift_fault(tstart, sensor_idx, drift_rate=param), {"drift_rate": param}
    else:
        raise ValueError(f"Unknown fault type: {fault_type}")


# -----------------------------------------------------------------------------
# Single simulation run
# -----------------------------------------------------------------------------
def run_single_simulation(dt, fault_generators, detector_eps, S_list, sim_time=65.0):
    """
    Run a single simulation with the given list of fault generator functions.
    Return a pandas DataFrame with the time-series data.
    """

    # Plant
    x0 = np.array([100, 200, 0, 0, 0])
    plant = KinematicBicycle5StateRearWheelRefPlant(
        x0,
        dt,
        L=4.094,
        max_steer=0.59,
        max_speed=107.29,
        max_accel=3.3 * 9.81,
        max_steer_rate=np.inf,
    )
    sensor = KinematicBicycleRaceDaySensor()

    # Estimator
    x0_hat = x0 + np.array([2, 2, 1, 0.1, 0.01])
    ukf_noise_std = np.array([0.5, 0.5, 0.1, 0.5, 0.5, 0.1])
    P = np.diag([1, 1, 0.3, 0.5, 0.1])
    R = np.diag((ukf_noise_std) ** 2)
    Q = np.diag([0.1, 0.1, 0.01, 0.1, 0.001])
    estimator = SimpleUKF(plant.model, sensor, dt, x0_hat, P, R, Q)

    # Planner
    planner = StaticFigureEightPlanner(
        center=[0, 0],
        length=2000,
        width=1000,
        num_points=100000,
        target_velocity_fn=lambda _pts, _hdg, crv, _dK: calc_target_velocity(
            crv, plant.max_speed
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
    detector = LookAheadDetector(
        plant.model, sensor, N, dt, detector_eps, S_list
    )

    # Data storage
    x_list = []
    z_list = []
    validity_list = []
    x_hat_list = []
    u_list = []
    calc_validity_meta = []

    sim_steps = int(sim_time / dt)
    start_time = time.perf_counter()

    for k in range(sim_steps):
        state = plant.get_state()
        true_output = sensor.get_output(state)

        # Apply faults
        measured_output = true_output
        for fg in fault_generators:
            measured_output = fg(k, measured_output)

        # Detector, Estimator
        validity, cv_meta = detector.calc_validity()
        estimate = estimator.estimate(measured_output, plant.u, validity)

        # Plan + Control
        plan = planner.plan(estimate)
        inp, _ = controller.step(plan, estimate)

        # Advance detector and plant
        detector.step(measured_output, estimate, plan, inp)
        plant.set_inputs(inp)
        plant.next()

        # Store
        x_list.append(state)
        z_list.append(measured_output)
        validity_list.append(validity)
        x_hat_list.append(estimate)
        u_list.append(inp)
        calc_validity_meta.append(cv_meta)

    end_time = time.perf_counter()

    x_arr = np.array(x_list)
    x_hat_arr = np.array(x_hat_list)
    z_arr = np.array(z_list)
    validity_arr = np.array(validity_list)

    df = pd.DataFrame(
        {
            "t": np.arange(sim_steps) * dt,
            "x_true": x_arr[:, 0],
            "y_true": x_arr[:, 1],
            "theta_true": x_arr[:, 2],
            "v_true": x_arr[:, 3],
            "delta_true": x_arr[:, 4],
            "x_hat": x_hat_arr[:, 0],
            "y_hat": x_hat_arr[:, 1],
            "theta_hat": x_hat_arr[:, 2],
            "v_hat": x_hat_arr[:, 3],
            "delta_hat": x_hat_arr[:, 4],
            "z0_meas": z_arr[:, 0],
            "z1_meas": z_arr[:, 1],
            "z2_meas": z_arr[:, 2],
            "z3_meas": z_arr[:, 3],
            "z4_meas": z_arr[:, 4],
            "z5_meas": z_arr[:, 5],
            "detector_total_time_s": [m.total_time for m in calc_validity_meta],
            "detector_setup_time_s": [m.optimizer_metadata.setup_time if m.optimizer_metadata else np.nan for m in calc_validity_meta],
            "detector_solve_time_s": [m.optimizer_metadata.solve_time if m.optimizer_metadata else np.nan for m in calc_validity_meta],
        }
    )

    # validity flags
    for sensor_idx in range(validity_arr.shape[1]):
        df[f"valid_sensor_{sensor_idx}"] = validity_arr[:, sensor_idx]

    df["simulation_wall_time_s"] = end_time - start_time
    return df


# -----------------------------------------------------------------------------
# Main function to run multiple simulations
# -----------------------------------------------------------------------------
@app.command()
def run_multiple(
    out_file_template: Path = typer.Option(
        "results.csv", help="Template for the output file name."
    ),
    num_simulations: int = typer.Option(5, help="Number of simulations to run."),
    fault_type: str = typer.Option(
        "random",
        help="Choose fault type: " + ", ".join(FAULT_TYPES) + ", or random.",
    ),
    num_faulty_sensors: int | None = typer.Option(None, help="Number of sensors to fault. If not specified, randomly selects 1 or 2."),
    possible_faulty_sensors_str: str = typer.Option(
        "2,3,4,5", help="Comma-separated list of possible faulty sensors. Default is 2,3,4,5 (heading, velocity1, velocity2, steering) (position sensors 0 and 1 are protected.)"
    ),  
    fault_start_time: int | None = typer.Option(None, help="Fault start time in simulation steps. If not specified, randomly selects a start time."),
    dt: float = typer.Option(0.01, help="Time step (s)."),
    time_per_sim: float = typer.Option(65.0, help="Simulation time (s)."),
    eps_scaler: float = typer.Option(None, help="The scaler for eps. Randomly samples from U(1,10) if not specified"),
    random_seed: int = typer.Option(
        None,
        help="Random seed for reproducibility. If not specified, uses the current time.",
    ),
):
    """
    Run multiple simulations, each injecting faults of a specified or random type,
    and save all results in a single Parquet file.
    """
    all_possible_sensors = [0, 1, 2, 3, 4, 5]
    possible_faulty_sensors = [int(s) for s in possible_faulty_sensors_str.split(",")]
    # Ensure the sensors are valid
    if not set(possible_faulty_sensors).issubset(set(all_possible_sensors)):
        raise ValueError(
            f"Invalid sensors specified. Possible sensors are: {all_possible_sensors}. Provided: {possible_faulty_sensors}"
        )

    # Set random seed
    if random_seed is None:
        random_seed = int(time.time())
    random.seed(random_seed)
    print("Using random seed:", random_seed)

    for sim_idx in tqdm(range(num_simulations), desc="Simulations"):
        # Decide how many sensors to fault (1 or 2 for demonstration)
        if num_faulty_sensors is None:
            num_faulty_sensors = random.choice([1, 2])

        # Randomly pick sensors to fault
        faulty_sensors = random.sample(possible_faulty_sensors, k=num_faulty_sensors)

        # Build fault generator functions
        if fault_start_time is None:
            fault_start_time = random.randint(1, int(time_per_sim / dt))  # same random start time for all sensors
        fault_functions = []
        fault_types = []
        fault_params = []
        for sensor_idx in faulty_sensors:
            # This picks the user-specified (or random) fault type for each sensor
            if fault_type == "random":
                actual_fault_type = random.choice(list(FAULT_RANGES.keys()))
            else:
                actual_fault_type = fault_type
            fault_fn_factory, params = create_fault_fn(actual_fault_type, sensor_idx)
            fault_func = fault_fn_factory(fault_start_time)

            fault_types.append(actual_fault_type)
            fault_params.append(params)
            fault_functions.append(fault_func)

        # Determine detector eps
        if not eps_scaler:
            eps_scaler = random.uniform(0, 10)
        detector_eps = np.array([1.5,1.5,0.3,1.5,1.5,0.3]) * eps_scaler

        # Run simulation
        df_timeseries = run_single_simulation(
            dt,
            fault_functions,
            detector_eps,
            S_list=[s for s in powerset(all_possible_sensors) if (set(all_possible_sensors)-set(possible_faulty_sensors)).issubset(s)],
            sim_time=time_per_sim,
        )


        out_file = out_file_template.with_name(out_file_template.stem + f".{sim_idx}" + out_file_template.suffix)
        out_file_meta = out_file.with_suffix(".meta.json")

        # Tag metadata
        df_timeseries["sim_file"] = out_file.name

        # Store simulation parameters
        sim_metadata = {
            "sim_file": out_file.name,
            "eps_scaler": eps_scaler,
            "fault_start_time": fault_start_time,
            "num_faulty_sensors": num_faulty_sensors,
            "faulty_sensors": faulty_sensors,
            "fault_types": fault_types,
            "fault_params": fault_params,
        }

        if out_file.suffix == ".parquet":
            df_timeseries.to_parquet(out_file)
        elif out_file.suffix == ".csv":
            df_timeseries.to_csv(out_file, index=False)
        else:
            print(f"WARNING: Unknown file extension: {out_file.suffix}. Defaulting to Parquet.")
            df_timeseries.to_parquet(out_file.with_suffix(".parquet"))
        
        with open(out_file_meta, "w") as f:
            json.dump(sim_metadata, f)

    typer.echo(f"Saved {num_simulations} simulations with template {str(out_file_template)}.")

@app.command()
def post_process(results_files: List[Path] = typer.Argument(..., callback=expand_glob, help="Path to results file(s).")):
    dfs = []
    for results_file in tqdm(results_files, total=len(results_files), desc="Reading result files"):
        if results_file.suffix == ".parquet":
            df = pd.read_parquet(results_file)
        elif results_file.suffix == ".csv":
            df = pd.read_csv(results_file)
        
        if "sim_id" not in df.columns:
            print("WARNING: sim_id not found in DataFrame. Treating as a single simulation.")
            df["sim_id"] = f"{results_file.stem}-0"
        else:
            df["sim_id"] = f"{results_file.stem}-" + df['sim_id'].astype(str)
        
        dfs.append(df)

    _post_process(pd.concat(dfs, ignore_index=True))

def _post_process(df_runs):
    validity_columns = [col for col in df_runs.columns if col.startswith("valid_sensor_")]

    grouped = df_runs.groupby("sim_id")

    detected_faults = grouped[validity_columns].all().apply(lambda x: not x.all(), axis=1)

    assert (grouped["eps_scaler"].nunique() == 1).all(), f"Multiple eps_scalers detected: {grouped['eps_scaler'].unique()}"
    eps_scalers = grouped["eps_scaler"].first()

    assert (grouped["fault_start_time"].nunique() == 1).all(), f"Multiple fault_start_times detected: {grouped['fault_start_time'].unique()}"
    fault_start_times = grouped["fault_start_time"].first()

    num_faulty_sensors = grouped["num_faulty_sensors"].first()
    faulty_sensor_0 = grouped["faulty_sensor_0"].first()
    fault_params_0 = grouped["fault_params_0"].first()

    # fault_types = []
    # fault_params = []
    # faulty_sensors = []
    # for i in range(num_faulty_sensors):
    #     if f"faulty_sensor_{i}" in df_runs.columns:
    #         faulty_sensors.append(grouped[f"faulty_sensor_{i}"].first())
    #         fault_types.append(grouped[f"fault_type_{i}"].first())
    #         fault_params.append(grouped[f"fault_params_{i}"].first())
    #     else:
    #         faulty_sensors.append(None)
    #         fault_types.append(None)
    #         fault_params.append(None)

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    app()
