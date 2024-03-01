# %%
# Configure auto-reload for imports
#! echo "If you see this in the output, magic commands are correctly configured."
# Automatically reload modules
#! %load_ext autoreload
#! %autoreload 2

# %%
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
from slugify import slugify

sys.path.append(str(Path(__file__).parent.parent))

from analysis.utils import (
    load_and_prepare_data,
    plot_confusion_matrix,
    plot_fault_distribution,
    plot_detection_delay,
    plot_generic_detection_data,
    calculate_and_plot_detection_percentage,
)

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
np.set_printoptions(suppress=True)

# %%

exp_path = Path(__file__).parent.parent.parent / "exp"

# file_path = exp_path / "test.jsonl"
# fault_conf_column = "fault_spec.kwargs.bias"
# fault_name = "Bias"
# exp_names = ["fine-grained-bias-sweep-4", "fine-grained-bias-sweep-3"]
# load_and_prepare_data(file_path, exp_names) # preload data into cache

# file_path = exp_path / "test-bias-angular.jsonl"
# fault_conf_column = "fault_spec.kwargs.bias"
# fault_name = "Bias"
# exp_names = []
# load_and_prepare_data(file_path, exp_names) # preload data into cache

# file_path = exp_path / "test-noise.jsonl"
# fault_conf_column = "fault_spec.kwargs.noise_level"
# fault_name = "Noise Level"
# exp_names = []
# load_and_prepare_data(file_path, exp_names) # preload data into cache

# file_path = exp_path / "test-noise-steering.jsonl"
# fault_conf_column = "fault_spec.kwargs.noise_level"
# fault_name = "Noise Level"
# exp_names = []
# load_and_prepare_data(file_path, exp_names) # preload data into cache

# file_path = exp_path / "test-spike.jsonl"
# fault_conf_column = "fault_spec.kwargs.spike_value"
# fault_name = "Spike value"
# exp_names = []
# load_and_prepare_data(file_path, exp_names) # preload data into cache

# file_path = exp_path / "test-spike.jsonl"
# fault_conf_column = "fault_spec.kwargs.duration"
# fault_name = "Spike duration"
# exp_names = []
# load_and_prepare_data(file_path, exp_names) # preload data into cache

# file_path = exp_path / "test-spike-steering.jsonl"
# fault_conf_column = "fault_spec.kwargs.spike_value"
# fault_name = "Spike value"
# exp_names = []
# load_and_prepare_data(file_path, exp_names) # preload data into cache

# file_path = exp_path / "test-spike-steering.jsonl"
# fault_conf_column = "fault_spec.kwargs.duration"
# fault_name = "Spike duration"
# exp_names = []
# load_and_prepare_data(file_path, exp_names) # preload data into cache

# file_path = exp_path / "test-drift-sensors-2-3.jsonl"
# fault_conf_column = "fault_spec.kwargs.drift_rate"
# fault_name = "Drift Rate"
# exp_names = []
# load_and_prepare_data(file_path, exp_names) # preload data into cache

# file_path = exp_path / "test-drift-sensors-4-5.jsonl"
# fault_conf_column = "fault_spec.kwargs.drift_rate"
# fault_name = "Drift Rate"
# exp_names = []
# load_and_prepare_data(file_path, exp_names) # preload data into cache

# file_path = exp_path / "comprehensive-nebula-new.jsonl"
# fault_fn = "sensor_bias_fault"
# fault_conf_column = "fault_spec.kwargs.bias"
# fault_name = "Bias"
# exp_names = []
# load_and_prepare_data(file_path, exp_names) # preload data into cache

# file_path = exp_path / "comprehensive-racecar.jsonl"
file_path = exp_path / "randomized-fixed.jsonl"

# bias
# fault_fn = "sensor_bias_fault"
# fault_conf_column = "fault_spec.kwargs.bias"
# fault_name = "Bias"

# spike value
# fault_fn = "spike_fault"
# fault_conf_column = "fault_spec.kwargs.spike_value"
# fault_name = "Spike value"

# spike duration
# fault_fn = "spike_fault"
# fault_conf_column = "fault_spec.kwargs.duration"
# fault_name = "Spike duration"

# noise
fault_fn = "random_noise_fault"
fault_conf_column = "fault_spec.kwargs.noise_level"
fault_name = "Noise Level"

# drift
# fault_fn = "drift_fault"
# fault_conf_column = "fault_spec.kwargs.drift_rate"
# fault_name = "Drift Rate"



sensor_indices = []

exp_names = []

# %%
df_all = load_and_prepare_data(file_path, exp_names)
# Select only the data with the specified fault
df = df_all.loc[(df_all["fault_spec.fn"] == fault_fn) | (df_all["fault_spec.fn"] == "noop")]

df_fault = df.loc[df["fault_spec.fn"] != "noop"]

# %%
# Plots

plot_confusion_matrix(df)
# plot_fault_distribution(df_fault, fault_conf_column, fault_name)

# Analysis for each sensor
for sensor_idx in sensor_indices or sorted(df_fault["fault_spec.kwargs.sensor_idx"].unique()):
    sensor_data = df_fault[df_fault["fault_spec.kwargs.sensor_idx"] == sensor_idx]

    print(f"Detection delay for sensor {sensor_idx}")
    plot_detection_delay(sensor_data, sensor_idx, fault_name, fault_conf_column)
    plt.title("")
    plt.savefig(slugify(f"{file_path.stem}_detection_delay_{fault_name}_sensor_{sensor_idx}") + ".pdf", format="pdf")
    plt.show()

    print(f"Detection percentage for sensor {sensor_idx}")
    calculate_and_plot_detection_percentage(df_fault, sensor_idx, fault_name, fault_conf_column)
    plt.title("")
    plt.savefig(slugify(f"{file_path.stem}_detection_percentage_{fault_name}_sensor_{sensor_idx}") + ".pdf", format="pdf")
    plt.show()

# %%
