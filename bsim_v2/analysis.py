# %%
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
import time
import hashlib
import os


def sha256sum(filename):
    """
    Calculate the SHA256 checksum of a file.
    Derived from https://stackoverflow.com/a/44873382
    """
    with open(filename, "rb", buffering=0) as f:
        return hashlib.file_digest(f, "sha256").hexdigest()  # type: ignore


def preprocess_jsonl(filename):
    """
    Preprocess a JSONL file by running pd.json_normalize on each line and
    returning a stream of DataFrames.
    """
    with open(filename, "r") as file:
        for line in file:
            json_obj = json.loads(line)
            yield pd.json_normalize(json_obj)


def load_data(filename):
    """
    Load a JSONL file into a Pandas DataFrame.
    """
    # Check if a cached version of the processed file exists
    cache_file = Path(f"/tmp/{os.getlogin()}/bsim_v2_cache/{sha256sum(filename)}.pkl")
    if cache_file.exists():
        return pd.read_pickle(cache_file)

    # otherwise, preprocess the file and save it
    data = pd.concat(preprocess_jsonl(filename), ignore_index=True)

    # Create the cache directory if it doesn't exist
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    data.to_pickle(cache_file)
    return data


# Replace null values in 'det_delay' with infinity
def prepare_data(df):
    # df["det_delay"].replace({np.nan: np.inf}, inplace=True)
    df["det_delay"] = df["det_delay"].fillna(np.inf)
    return df


# Plotting function for scatter and marker plots
def plot_sensor_data(sensor_data, sensor_idx):
    detected_data = sensor_data[sensor_data["det_delay"] < np.inf]
    not_detected_data = sensor_data[sensor_data["det_delay"] == np.inf]

    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        x="fault_spec.kwargs.bias", y="det_delay", data=detected_data, label="Detected"
    )
    plt.scatter(
        not_detected_data["fault_spec.kwargs.bias"],
        [max(detected_data["det_delay"]) + 0.01] * len(not_detected_data),
        color="red",
        marker="x",
        label="Not Detected",
    )
    plt.title(f"Detection Delay vs Bias for Sensor {sensor_idx}")
    plt.xlabel("Bias (fault_spec.kwargs.bias)")
    plt.ylabel("Detection Delay (det_delay)")
    plt.ylim(bottom=0)
    plt.legend()
    plt.show()


# Function to calculate and plot detection percentage
def calculate_and_plot_detection_percentage(df, sensor_idx):
    sensor_data = df[df["fault_spec.kwargs.sensor_idx"] == sensor_idx]
    detection_percentage = sensor_data.groupby("fault_spec.kwargs.bias")[
        "det_delay"
    ].apply(lambda x: (x < np.inf).mean() * 100)

    plt.figure(figsize=(12, 6))
    detection_percentage.plot(kind="bar")
    plt.title(f"Percentage of Successful Detections vs Bias for Sensor {sensor_idx}")
    plt.xlabel("Bias (fault_spec.kwargs.bias)")
    plt.ylabel("Percentage of Successful Detections (%)")

    tick_labels = detection_percentage.index
    total_ticks = len(tick_labels)
    num_ticks = 11
    tick_spacing = max(1, total_ticks // num_ticks)
    selected_ticks = tick_labels[::tick_spacing]
    plt.xticks(range(0, total_ticks, tick_spacing), [f"{label:.2f}" for label in selected_ticks], rotation=45)

    plt.show()

# %%

exp_path = Path(__file__).parent.parent / "exp"
file_path = exp_path / "test.jsonl"

# Main analysis
print("Loading data...")
start = time.perf_counter()
df_pd = load_data(file_path)
print(f"Data loaded in {time.perf_counter() - start:.2f} seconds")


# %%

print("Preparing data...")
start = time.perf_counter()
df = prepare_data(df_pd)
# use only specific experiments
df = df[df["exp_name"].isin(["fine-grained-bias-sweep-4", "fine-grained-bias-sweep-3"])]
print(f"Data prepared in {time.perf_counter() - start:.2f} seconds")

# Analysis for each sensor
for sensor_idx in df["fault_spec.kwargs.sensor_idx"].unique():
    sensor_data = df[df["fault_spec.kwargs.sensor_idx"] == sensor_idx]
    plot_sensor_data(sensor_data, sensor_idx)
    calculate_and_plot_detection_percentage(df, sensor_idx)

# %%
