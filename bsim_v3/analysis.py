import marimo

__generated_with = "0.10.19"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    from tqdm import tqdm
    import pandas as pd
    return Path, mo, pd, tqdm


@app.cell
def _():
    import json
    return (json,)


@app.cell
def _():
    from copy import deepcopy
    return (deepcopy,)


@app.cell
def _(Path):
    BASE_PATH = Path("exp/bsim_v3/sweep-1")
    return (BASE_PATH,)


@app.cell
def _():
    dt = 0.01 # Must be consistent with the sim.
    return (dt,)


@app.cell
def _(BASE_PATH, json, tqdm):
    meta_files = list(BASE_PATH.glob("*.meta.json"))
    raw_metas = [json.load(open(meta_file)) for meta_file in tqdm(meta_files, desc="Loading meta files", total=len(meta_files))]
    return meta_files, raw_metas


@app.cell
def _(raw_metas):
    # Sample a metadata
    raw_metas[0]
    return


@app.cell
def _(deepcopy, raw_metas):
    # Preprocess metas to flatten the nested lists
    processed_metas = []

    for raw_meta in raw_metas:
        meta = deepcopy(raw_meta)
        for i in range(meta["num_faulty_sensors"]):
            # meta[f"has_fault_on_sensor_{meta['faulty_sensors'][i]}"] = True
            meta[f"fault_{i}_sensor"] = meta["faulty_sensors"][i]
            meta[f"fault_{i}_type"] = meta["fault_types"][i]
            for param, value in meta["fault_params"][i].items():
                meta[f"fault_{i}_{param}"] = value

        del meta["faulty_sensors"]
        del meta["fault_types"]
        del meta["fault_params"]

        processed_metas.append(meta)

        # # TODO: remove this after debugging
        # if len(processed_metas) >= 100:
        #     break
    return i, meta, param, processed_metas, raw_meta, value


@app.cell
def _(dt, pd, processed_metas):
    metas_df = pd.DataFrame(processed_metas).set_index("sim_file")
    metas_df["fault_start_time_s"] = metas_df["fault_start_time"] * dt
    metas_df
    return (metas_df,)


@app.cell
def _(BASE_PATH, metas_df, pd):
    sims_df = pd.read_parquet([BASE_PATH / f for f in metas_df.index])
    return (sims_df,)


@app.cell
def _(sims_df):
    sims_df[:100]
    return


@app.cell
def _(sims_df):
    len(sims_df)
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Fault detection performance

        For each simulation, we gather the following data:

        - fault metadata (sensor set, time)
        - detection data
            - First detected fault (sensor set and time)

        Then look at set-based precision and recall (compare the actual faulty set to the detected faulty set).

        References:

        - https://chatgpt.com/share/679f7571-cfa0-8010-8e16-77b116482e7f
        """
    )
    return


@app.cell
def _(sims_df):
    validity_columns = [col for col in sims_df.columns if col.startswith("valid_sensor_")]
    validity_columns
    return (validity_columns,)


@app.cell
def _(sims_df):
    grouped = sims_df.groupby("sim_file")
    return (grouped,)


@app.cell
def _(grouped, validity_columns):
    # Whether each sensor is valid during the entire simulation
    grouped[validity_columns].all()
    return


@app.cell
def _(metas_df, pd, validity_columns):
    def process_sim(group):
        sim_file = group.iloc[0]["sim_file"]
        fault_start_idx = metas_df.loc[sim_file, "fault_start_time"]
        fault_start_x = group.iloc[fault_start_idx]["x_true"]
        fault_start_y = group.iloc[fault_start_idx]["y_true"]
        fault_start_v = group.iloc[fault_start_idx]["v_true"]

        exists_invalid_sensors = (group[validity_columns] == False).any(axis=1) # whether each time step (row) has invalid sensors
        if exists_invalid_sensors.any():
            first_invalid_index = exists_invalid_sensors.idxmax()  # Get index of first invalid occurrence
            first_invalid_row = group.loc[first_invalid_index]

            ret = pd.Series({
                "time_of_detection": first_invalid_row["t"],
                "is_fault_detected": True,
                "fault_start_x": fault_start_x,
                "fault_start_y": fault_start_y,
                "fault_start_v": fault_start_v,
                **first_invalid_row[validity_columns].to_dict(),
            })
        else:
            ret = pd.Series({
                "time_of_detection": None,
                "is_fault_detected": False,
                "fault_start_x": fault_start_x,
                "fault_start_y": fault_start_y,
                "fault_start_v": fault_start_v,
                **{col: None for col in validity_columns},
            })
        return ret
    return (process_sim,)


@app.cell
def _(grouped, process_sim):
    processed = grouped.apply(process_sim)
    processed
    return (processed,)


@app.cell
def _(metas_df, pd, processed):
    # Evaluate metrics
    results = pd.concat([processed, metas_df], axis="columns")
    results
    return (results,)


@app.cell
def _(results):
    # Save results to file
    results.to_csv("results.csv")
    return


@app.cell
def _(results_with_metrics):
    # Plot the effect of eps_scaler on fault detection
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=results_with_metrics, x="eps_scaler", hue="is_fault_detected", multiple="stack", bins=50)
    plt.xlabel("eps_scaler")
    plt.ylabel("Count")
    plt.title("Effect of eps_scaler on fault detection")
    plt.show()
    return plt, sns


@app.cell
def _(results_with_metrics):
    results_with_metrics["is_fault_detected"]==False
    return


@app.cell
def _(results_with_metrics):
    def _():
        # Plot the effect of eps_scaler on fault detection
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        sns.histplot(data=results_with_metrics[results_with_metrics["is_fault_detected"] == False], x="eps_scaler", hue="precision", multiple="stack", bins=50)
        plt.xlabel("eps_scaler")
        plt.ylabel("Count")
        plt.title("Effect of eps_scaler on precision")
        return plt.show()
    _()
    return


@app.cell
def _(results_with_metrics):
    def _():
        # Plot the effect of eps_scaler on fault detection
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        sns.histplot(data=results_with_metrics[results_with_metrics["is_fault_detected"] == False], x="eps_scaler", hue="recall", multiple="stack", bins=50)
        plt.xlabel("eps_scaler")
        plt.ylabel("Count")
        plt.title("Effect of eps_scaler on recall")
        return plt.show()
    _()
    return


@app.cell
def _(mo):
    mo.md(r"""When `eps_scaler` is too large, some faults don't get detected. The faults may be tolerated by the loose threshold.""")
    return


@app.cell
def _(mo):
    fault_sensor_dropdown = mo.ui.dropdown(
        options={str(sensor): sensor for sensor in range(6)},
        value="2",
        label="Pick a sensor",
    )
    fault_sensor_dropdown
    return (fault_sensor_dropdown,)


@app.cell
def _(fault_sensor_dropdown, plt, results, sns):
    # Plot the effects of fault bias on detection
    def _():
        filtered_results = results[(results["num_faulty_sensors"] == 1) & (results["fault_0_type"] == "bias") & (results['fault_0_sensor'] == fault_sensor_dropdown.value)]

        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        sns.histplot(data=filtered_results, x="fault_0_bias", hue="is_fault_detected", multiple="stack", bins=50)
        plt.xlabel("Fault Bias")
        plt.ylabel("Count")
        plt.title("Effect of Fault Bias on Detection")
        return plt.gca()
    _()

    # TODO: normalize the fault bias by sensor
    return


@app.cell
def _(pd, results, validity_columns):
    def calculate_metrics(sim_row, num_sensors):
        # Get ground truth faulty sensors from metadata
        num_faulty_sensors = sim_row['num_faulty_sensors']
        faulty_sensors = set(sim_row[[f'fault_{i}_sensor' for i in range(num_faulty_sensors)]])

        # Get detected sensors (where validity is False)
        detected_sensors = set()
        for col in validity_columns:
            if sim_row[col] is False:
                detected_sensors.add(int(col.split('_')[-1]))

        # Calculate true positives, false positives, false negatives, and true negatives
        tp = len(faulty_sensors.intersection(detected_sensors))
        fp = len(detected_sensors - faulty_sensors)
        fn = len(faulty_sensors - detected_sensors)
        tn = num_sensors - (tp + fp + fn)

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0 # aka "IoU"

        return pd.Series({
            'precision': precision,
            'recall': recall,
            'jaccard': jaccard,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'detected_sensors': detected_sensors,
            'faulty_sensors': faulty_sensors,
        })

    # Apply metrics calculation to each simulation
    metrics_df = results.apply(calculate_metrics, num_sensors=6, axis=1)
    results_with_metrics = pd.concat([results, metrics_df], axis=1)
    return calculate_metrics, metrics_df, results_with_metrics


@app.cell
def _(results_with_metrics):
    # save results_with_metrics to a file
    results_with_metrics.to_csv("results_with_metrics.csv")
    return


@app.cell
def _(results_with_metrics):
    results_with_metrics
    return


@app.cell
def _(results_with_metrics):
    results_with_metrics["detection_delay"] = results_with_metrics["fault_start_time_s"] - results_with_metrics["time_of_detection"]
    return


@app.cell
def _(results_with_metrics):
    results_with_metrics
    return


@app.cell
def _(results_with_metrics):
    results_with_metrics.groupby(["fault_0_type","fault_1_type"],dropna=False)["precision"].describe()
    return


@app.cell
def _(results_with_metrics):
    results_with_metrics[results_with_metrics["eps_scaler"]<0.5].groupby(["fault_0_type","fault_1_type"],dropna=False)["precision"].describe()
    return


@app.cell
def _(results_with_metrics):
    results_with_metrics.groupby(["fault_0_sensor","fault_1_sensor"],dropna=False)["recall"].describe()
    return


@app.cell
def _(results_with_metrics):
    results_with_metrics.groupby(["fault_0_sensor","fault_1_sensor"],dropna=False)["jaccard"].describe()
    return


@app.cell
def _():
    return


@app.cell
def _(plt, results_with_metrics, sns):
    def _():
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        sns.set_style("whitegrid")

        for sensor in range(6):
            # Filter data for current sensor
            sensor_data = results_with_metrics[
                (results_with_metrics[f'fault_0_sensor'] == sensor) |
                (results_with_metrics[f'fault_1_sensor'] == sensor)
            ]

            # Calculate metrics
            avg_precision = sensor_data['precision'].mean()
            avg_recall = sensor_data['recall'].mean()
            avg_jaccard = sensor_data['jaccard'].mean()
            count = len(sensor_data)

            # Plot metrics
            axes[0].bar(sensor, avg_precision)
            axes[1].bar(sensor, avg_recall)
            axes[2].bar(sensor, avg_jaccard)

            # Add count to bars
            axes[0].text(sensor, avg_precision + 0.01, f'n={count}', ha='center')
            axes[1].text(sensor, avg_recall + 0.01, f'n={count}', ha='center')
            axes[2].text(sensor, avg_jaccard + 0.01, f'n={count}', ha='center')

        axes[0].set_title('Precision')
        axes[1].set_title('Recall')
        axes[2].set_title('Jaccard Index')

        axes[0].set_xlabel('Sensor')
        axes[1].set_xlabel('Sensor')
        axes[2].set_xlabel('Sensor')

        plt.tight_layout()
        return plt.gca()

    _()
    return


if __name__ == "__main__":
    app.run()
