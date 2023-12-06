# %%
# Configure auto-reload for imports
#! echo "If you see this in the output, magic commands are correctly configured."
# Automatically reload modules
#! %load_ext autoreload
#! %autoreload 2

# %%
import matplotlib.pyplot as plt
from pathlib import Path
import time
import sys

sys.path.append(str(Path(__file__).parent.parent))

from analysis.utils import (
    load_data,
    prepare_data,
    plot_confusion_matrix,
    plot_sensor_data,
    calculate_and_plot_detection_percentage,
)

# %%

exp_path = Path(__file__).parent.parent.parent / "exp"

# file_path = exp_path / "test.jsonl"
# fault_conf_column = "fault_spec.kwargs.bias"
# fault_name = "Bias"
# exp_names = ["fine-grained-bias-sweep-4", "fine-grained-bias-sweep-3"]

# file_path = exp_path / "test-bias-angular.jsonl"
# fault_conf_column = "fault_spec.kwargs.bias"
# fault_name = "Bias"
# exp_names = []

# file_path = exp_path / "test-noise.jsonl"
# fault_conf_column = "fault_spec.kwargs.noise_level"
# fault_name = "Noise Level"
# exp_names = []

# file_path = exp_path / "test-noise-steering.jsonl"
# fault_conf_column = "fault_spec.kwargs.noise_level"
# fault_name = "Noise Level"
# exp_names = []

# file_path = exp_path / "test-spike.jsonl"
# fault_conf_column = "fault_spec.kwargs.spike_value"
# fault_name = "Spike value"
# exp_names = []

# file_path = exp_path / "test-spike.jsonl"
# fault_conf_column = "fault_spec.kwargs.duration"
# fault_name = "Spike duration"
# exp_names = []

# file_path = exp_path / "test-spike-steering.jsonl"
# fault_conf_column = "fault_spec.kwargs.spike_value"
# fault_name = "Spike value"
# exp_names = []

# file_path = exp_path / "test-spike-steering.jsonl"
# fault_conf_column = "fault_spec.kwargs.duration"
# fault_name = "Spike duration"
# exp_names = []

# file_path = exp_path / "test-drift-sensors-2-3.jsonl"
# fault_conf_column = "fault_spec.kwargs.drift_rate"
# fault_name = "Drift Rate"
# exp_names = []

file_path = exp_path / "test-drift-sensors-4-5.jsonl"
fault_conf_column = "fault_spec.kwargs.drift_rate"
fault_name = "Drift Rate"
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
