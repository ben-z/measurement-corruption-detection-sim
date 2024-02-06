# %%
# Configure auto-reload for imports
#! echo "If you see this in the output, magic commands are correctly configured."
# Automatically reload modules
#! %load_ext autoreload
#! %autoreload 2

# %%
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from analysis.utils import (
    load_and_prepare_data,
    plot_confusion_matrix,
    plot_fault_distribution,
    plot_detection_delay,
    plot_generic_detection_data,
    calculate_and_plot_detection_percentage,
)

# %%

exp_path = Path(__file__).parent.parent.parent / "exp"

# file_path = exp_path / "test-drift-sensors-2-3.jsonl"
# fault_conf_column = "fault_spec.kwargs.drift_rate"
# fault_name = "Drift Rate"
# exp_names = []
# load_and_prepare_data(file_path, exp_names) # preload data into cache

file_path = exp_path / "test-drift-sensors-4-5.jsonl"
fault_conf_column = "fault_spec.kwargs.drift_rate"
fault_name = "Drift Rate"
exp_names = []
load_and_prepare_data(file_path, exp_names) # preload data into cache

# %%
df = load_and_prepare_data(file_path, exp_names)

df_fault = df.loc[df["fault_spec.fn"] != "noop"]

# %%
plot_confusion_matrix(df)
plot_fault_distribution(df_fault, fault_conf_column, fault_name)

# %%
# Analysis for each sensor
for sensor_idx in df_fault["fault_spec.kwargs.sensor_idx"].unique():
    sensor_data = df_fault[df_fault["fault_spec.kwargs.sensor_idx"] == sensor_idx]
    sensor_data["det_magnitude"] = abs(sensor_data["fault_spec.kwargs.drift_rate"]) * sensor_data["det_delay"]
    plot_generic_detection_data(sensor_data, sensor_idx, fault_name, fault_conf_column, "Detected at Magnitude", "det_magnitude")
    plot_detection_delay(sensor_data, sensor_idx, fault_name, fault_conf_column)
    calculate_and_plot_detection_percentage(df_fault, sensor_idx, fault_name, fault_conf_column)
