# %%
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import seaborn as sns


# Loading the data
def load_data(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            json_obj = json.loads(line)
            flattened_data = pd.json_normalize(json_obj)
            data.append(flattened_data)
    return pd.concat(data, ignore_index=True)


# Replace null values in 'det_delay' with infinity
def prepare_data(df):
    df["det_delay"].replace({np.nan: np.inf}, inplace=True)
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
    labels = [f"{label:.2f}" for label in detection_percentage.index]
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.show()

# %%

def main():
    exp_path = pathlib.Path(__file__).parent.parent / "exp"
    file_path = exp_path / "test.jsonl"

    # data = []
    # with open(file_path, "r") as file:
    #     for line in file:
    #         # Parse the JSON string
    #         json_obj = json.loads(line)
    #         # Flatten the JSON object and append to the list
    #         flattened_data = pd.json_normalize(json_obj)
    #         data.append(flattened_data)

    # # Concatenate all flattened data into a single DataFrame
    # df = pd.concat(data, ignore_index=True)

    # # Display the DataFrame
    # print(df)

    # Main analysis
    df = load_data(file_path)
    df = prepare_data(df)

    # Analysis for each sensor
    for sensor_idx in df["fault_spec.kwargs.sensor_idx"].unique():
        sensor_data = df[df["fault_spec.kwargs.sensor_idx"] == sensor_idx]
        plot_sensor_data(sensor_data, sensor_idx)
        calculate_and_plot_detection_percentage(df, sensor_idx)

main()