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

def has_fault(df):
    ret = ~(df["fault_spec.fn"] == "noop")
    if "fault_spec.kwargs.bias" in df.columns:
        ret &= ~((df["fault_spec.fn"] == "sensor_bias_fault") & np.isclose(df["fault_spec.kwargs.bias"], 0))
    if "fault_spec.kwargs.noise_level" in df.columns:
        ret &= ~((df["fault_spec.fn"] == "random_noise_fault") & np.isclose(df["fault_spec.kwargs.noise_level"], 0))

    return ret

def prepare_data(df):
    # Drop legacy columns
    df = df.drop(columns=["det_delay"], errors="ignore")
    df["has_detection"] = df["corruption.t"].notnull()

    # Process data with fault and without fault separately
    prepare_data_with_fault(df, has_fault(df))

    return df

def prepare_data_with_fault(df, cond):
    df.loc[cond & df["has_detection"], "successful_detection"] = df["fault_spec.kwargs.start_t"] <= df["corruption.t"]
    df.loc[cond, "det_delay"] = df["corruption.t"] - df["fault_spec.kwargs.start_t"]
    df.loc[cond, "det_delay"] = df["det_delay"].fillna(np.inf)
    df.loc[cond, "det_delay"] = df["det_delay"].round(2)

def plot_confusion_matrix(df):
    """
    Plot the confusion matrix for a DataFrame.
    """

    # True Positive: fault present and detected
    # True Negative: fault not present and not detected
    # False Positive: fault not present but detected
    # False Negative: fault present but not detected
    # TP = df["successful_detection"].sum()
    TP = df[has_fault(df) & df["successful_detection"]].shape[0]
    TN = df[~has_fault(df) & ~df["has_detection"]].shape[0]
    FP = df[~has_fault(df) & df["has_detection"]].shape[0]
    FN = df[has_fault(df) & ~(df["successful_detection"].fillna(False))].shape[0]

    print(f"{TP=} {TN=} {FP=} {FN=}")

    cm_df = pd.DataFrame(
        [
            [TP, FN],
            [FP, TN]
        ],
        index=["Fault Present", "Fault Not Present"],
        columns=["Fault Detected", "Fault Not Detected"],
    )

    # Plot the confusion matrix
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        # [[TP, FP], [FN, TN]],
        cm_df,
        annot=True,
        fmt="d",
        # xticklabels=["Fault Detected", "Fault Not Detected"],
        # yticklabels=["Fault Present", "Fault Not Present"],
    )
    plt.title("Confusion Matrix")
    plt.show()

# Plotting function for scatter and marker plots
def plot_sensor_data(sensor_data, sensor_idx, fault_name, fault_conf_column):
    detected_data = sensor_data[sensor_data["det_delay"] < np.inf]
    not_detected_data = sensor_data[sensor_data["det_delay"] == np.inf]

    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        x=fault_conf_column, y="det_delay", data=detected_data, label="Detected"
    )
    plt.scatter(
        not_detected_data[fault_conf_column],
        [max(detected_data["det_delay"]) + 0.01] * len(not_detected_data),
        color="red",
        marker="x",
        label="Not Detected",
    )
    plt.title(f"Detection Delay vs {fault_name} for Sensor {sensor_idx}")
    plt.xlabel(f"{fault_name} ({fault_conf_column})")
    plt.ylabel("Detection Delay (det_delay)")
    plt.ylim(bottom=0)
    plt.legend()
    plt.show()


# Function to calculate and plot detection percentage
def calculate_and_plot_detection_percentage(df, sensor_idx, fault_name, fault_conf_column):
    sensor_data = df[df["fault_spec.kwargs.sensor_idx"] == sensor_idx]
    detection_percentage = sensor_data.groupby(fault_conf_column)[
        "det_delay"
    ].apply(lambda x: (x < np.inf).mean() * 100)

    plt.figure(figsize=(12, 6))
    detection_percentage.plot(kind="bar")
    plt.title(f"Percentage of Successful Detections vs {fault_name} for Sensor {sensor_idx}")
    plt.xlabel(f"{fault_name} ({fault_conf_column})")
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
# file_path = exp_path / "test.jsonl"
# fault_conf_column = "fault_spec.kwargs.bias"
# fault_name = "Bias"
# exp_names = ["fine-grained-bias-sweep-4", "fine-grained-bias-sweep-3"]

# file_path = exp_path / "test-noise.jsonl"
# fault_conf_column = "fault_spec.kwargs.noise_level"
# fault_name = "Noise Level"
# exp_names = []

# file_path = exp_path / "test-spike.jsonl"
# fault_conf_column = "fault_spec.kwargs.spike_value"
# fault_name = "Spike value"
# exp_names = []

file_path = exp_path / "test-spike.jsonl"
fault_conf_column = "fault_spec.kwargs.duration"
fault_name = "Spike duration"
exp_names = []

# Main analysis
print("Loading data...")
start = time.perf_counter()
df_raw = load_data(file_path)
print(f"Data loaded in {time.perf_counter() - start:.2f} seconds")


# %%

print("Preparing data...")
start = time.perf_counter()
# use only specific experiments
df = df_raw
if exp_names:
    df = df[df["exp_name"].isin(exp_names)]
df = prepare_data(df)
print(f"Data prepared in {time.perf_counter() - start:.2f} seconds")

df_fault = df.loc[df["fault_spec.fn"] != "noop"]

# df_fault["has_detection"].value_counts().plot(kind='barh')
# df_fault["successful_detection"].value_counts().plot(kind='barh')

plot_confusion_matrix(df)

df_fault.plot.hist(
    column=[fault_conf_column],
    by="fault_spec.kwargs.sensor_idx",
    bins=max(
        df_fault.groupby("fault_spec.kwargs.sensor_idx")[fault_conf_column].nunique()
    ),
    title=f"{fault_name} Distribution for Each Sensor",
)
plt.tight_layout()
plt.show()

# Analysis for each sensor
for sensor_idx in df_fault["fault_spec.kwargs.sensor_idx"].unique():
    sensor_data = df_fault[df_fault["fault_spec.kwargs.sensor_idx"] == sensor_idx]
    plot_sensor_data(sensor_data, sensor_idx, fault_name, fault_conf_column)
    calculate_and_plot_detection_percentage(df_fault, sensor_idx, fault_name, fault_conf_column)

# %%
