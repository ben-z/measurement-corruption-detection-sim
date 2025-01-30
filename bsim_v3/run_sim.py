import os
import time
import random
import numpy as np
import pandas as pd
import typer
from tqdm import tqdm

# ----- Local imports (your existing modules) -----
from lib.controllers.pure_pursuit import KinematicBicycle5StatePurePursuitController
from lib.detectors.detector import LookAheadDetector, CalcValidityMetadata
from lib.estimators.simple_ukf import SimpleUKF
from lib.fault_generators import (
    sensor_bias_fault,
    spike_fault,
    random_noise_fault,
    # if you have an intermittent_fault or complete_failure in your pipeline,
    # you can import them as well
)
from lib.planners.static import StaticFigureEightPlanner
from lib.planners.utils import calc_target_velocity
from lib.plants.kinematic_bicycle import KinematicBicycle5StateRearWheelRefPlant
from lib.sensors.kinematic_bicycle_race_day import KinematicBicycleRaceDaySensor

# ----- CLI setup with Typer -----
app = typer.Typer(
    help="Run multiple experiments with a chosen fault type and save to Parquet."
)

# ----- Disable multithreading for numpy (optional) -----
NUM_THREADS = "1"
os.environ["OMP_NUM_THREADS"] = NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS
os.environ["MKL_NUM_THREADS"] = NUM_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS


# -----------------------------------------------------------------------------
# Custom drift fault, if needed
# -----------------------------------------------------------------------------
def drift_fault(tstart, sensor_idx, drift_rate):
    """
    A fault function that adds a linearly increasing offset
    (i.e., drift) to the given sensor after tstart.
    drift = drift_rate * (k - tstart) for k >= tstart.
    """

    def fault_fn(k, z):
        if k < tstart:
            return z
        corrupted = z.copy()
        steps_since_start = k - tstart
        corrupted[sensor_idx] += drift_rate * steps_since_start
        return corrupted

    return fault_fn


# -----------------------------------------------------------------------------
# Ranges for random fault parameter generation
# -----------------------------------------------------------------------------
FAULT_RANGES = {
    "bias": {
        "heading": (-np.pi / 4, np.pi / 4),
        "velocity": (-5, 5),
        "steering": (-np.pi / 8, np.pi / 8),
    },
    "spike": {
        "heading": (-np.pi / 4, np.pi / 4),
        "velocity": (-5, 5),
        "steering": (-np.pi / 8, np.pi / 8),
    },
    "noise": {
        "heading": (0, np.pi / 4),  # amplitude
        "velocity": (0, 5),  # amplitude
        "steering": (0, np.pi / 8),  # amplitude
    },
    "drift": {
        "heading": (-np.pi / 2, np.pi / 2),  # rad/s
        "velocity": (-15, 15),  # m/s^2
        "steering": (-np.pi / 16, np.pi / 16),  # rad/s
    },
}

# Direct mapping from sensor_idx to the type in FAULT_RANGES
PARAM_RANGE_KEY = {
    0: None,  # x-position (no injection or treat as steering if you prefer)
    1: None,  # y-position
    2: "heading",
    3: "velocity",
    4: "velocity",
    5: "steering",
}


def create_fault_fn(fault_type, sensor_idx):
    """
    Given a fault type (bias, spike, noise, drift) and a sensor index,
    return a function that, given tstart, produces the final fault function.
    If `fault_type` is "random", pick from {bias, spike, noise, drift}.
    """
    if fault_type == "random":
        fault_type = random.choice(["bias", "spike", "noise", "drift"])

    # Determine the parameter-range key from the dictionary
    fkey = PARAM_RANGE_KEY.get(sensor_idx)
    # If None, fallback to "steering" or skip entirely.
    if fkey is None:
        raise ValueError(f"Sensor {sensor_idx} does not have a fault parameter.")

    low, high = FAULT_RANGES[fault_type][fkey]
    param = random.uniform(low, high)

    if fault_type == "bias":
        return lambda tstart: sensor_bias_fault(tstart, sensor_idx, bias=param)
    elif fault_type == "spike":
        duration = random.randint(1, 10)
        return lambda tstart: spike_fault(
            tstart, sensor_idx, amplitude=param, duration=duration
        )
    elif fault_type == "noise":
        return lambda tstart: random_noise_fault(tstart, sensor_idx, amplitude=param)
    elif fault_type == "drift":
        return lambda tstart: drift_fault(tstart, sensor_idx, drift_rate=param)
    else:
        raise ValueError(f"Unknown fault type: {fault_type}")


# -----------------------------------------------------------------------------
# Single simulation run
# -----------------------------------------------------------------------------
def run_single_simulation(dt, fault_generators, detector_eps, sim_time=65.0):
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
    detector = LookAheadDetector(plant.model, sensor, N, dt, detector_eps)

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
        }
    )

    # validity flags
    for sensor_idx in range(validity_arr.shape[1]):
        df[f"valid_sensor_{sensor_idx}"] = validity_arr[:, sensor_idx]

    df["simulation_wall_time_s"] = end_time - start_time
    return df


# -----------------------------------------------------------------------------
# Main function to run multiple experiments
# -----------------------------------------------------------------------------
@app.command()
def run_experiments(
    out_file: str = typer.Option(
        "results.csv", help="Path to output file."
    ),
    num_experiments: int = typer.Option(5, help="Number of experiments to run."),
    fault_type: str = typer.Option(
        "random",
        help="Choose fault type: 'bias', 'spike', 'noise', 'drift', or 'random'.",
    ),
    dt: float = typer.Option(0.01, help="Time step (s)."),
    time_per_sim: float = typer.Option(65.0, help="Simulation time (s)."),
):
    """
    Run multiple experiments, each injecting faults of a specified or random type,
    and save all results in a single Parquet file.
    """

    # We will store each experiment's DataFrame in a list
    all_dfs = []

    for exp_id in tqdm(range(num_experiments), desc="Experiments"):
        # Decide how many sensors to fault (1 or 2 for demonstration)
        num_faulty_sensors = random.choice([1, 2])

        # Randomly pick sensors to fault
        possible_sensors = [2, 3, 4, 5]  # heading, velocity1, velocity2, steering
        faulty_sensors = random.sample(possible_sensors, k=num_faulty_sensors)

        # Build fault generator functions
        fault_functions = []
        for sensor_idx in faulty_sensors:
            # This picks the user-specified (or random) fault type for each sensor
            fault_fn_factory = create_fault_fn(fault_type, sensor_idx)
            fault_start_time = random.randint(1, int(time_per_sim / dt))  # random start time
            fault_func = fault_fn_factory(fault_start_time)
            fault_functions.append(fault_func)

        # Determine detector eps
        eps_scaler = random.uniform(0, 1)
        detector_eps = np.array([1.5,1.5,0.3,1.5,1.5,0.3]) * eps_scaler

        # Run simulation
        df_run = run_single_simulation(dt, fault_functions, detector_eps, sim_time=time_per_sim)

        # Tag metadata
        df_run["experiment_id"] = exp_id
        df_run["faulty_sensors"] = str(faulty_sensors)
        df_run["fault_type"] = fault_type
        df_run["eps_scaler"] = eps_scaler

        all_dfs.append(df_run)

    # Combine all experiments into a single DataFrame
    df_all = pd.concat(all_dfs, ignore_index=True)

    # Save to Parquet
    if out_file.endswith(".parquet"):
        df_all.to_parquet(out_file)
    elif out_file.endswith(".csv"):
        df_all.to_csv(out_file, index=False)
    typer.echo(f"Saved {num_experiments} experiments to {out_file}")


def main():
    app()


if __name__ == "__main__":
    main()
