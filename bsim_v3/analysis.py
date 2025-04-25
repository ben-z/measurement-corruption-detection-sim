import marimo

__generated_with = "0.10.19"
app = marimo.App(width="full")


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
def _():
    import numpy as np
    return (np,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import seaborn as sns
    return plt, sns


@app.cell
def _():
    from IPython.display import display, HTML
    return HTML, display


@app.cell
def _(Path):
    # EXP_NAME = "sweep-5-fixed-eps"
    EXP_NAME = "sweep-6-higher-fault-range"
    # EXP_NAME = "bias-sweep-7"
    # EXP_NAME = "8-holistic-sweep"
    # EXP_NAME = "9-holistic-sweep"
    # EXP_NAME = "10-holistic-sweep-loose-heading-fault-range"
    BASE_PATH = Path("exp/bsim_v3") / EXP_NAME
    return BASE_PATH, EXP_NAME


@app.cell
def _(Path):
    OUTPUT_DIR = Path("visualizations/data/bsim_v3")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return (OUTPUT_DIR,)


@app.cell
def _():
    dt = 0.01 # Must be consistent with the sim.
    return (dt,)


@app.cell
def _(BASE_PATH, json, tqdm):
    # [::4] is for thinning out the meta files
    # meta_files = list(BASE_PATH.glob("*.meta.json"))[::8]

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
    metas_df[:100]
    return (metas_df,)


@app.cell
def _(metas_df):
    metas_df.index[0]
    return


@app.cell
def _(BASE_PATH, metas_df, pd, tqdm):
    # Parquet loading
    if metas_df.index[0].endswith(".parquet"):
        sims_df = pd.read_parquet([BASE_PATH / f for f in metas_df.index])
    # CSV loading
    elif metas_df.index[0].endswith(".csv"):
        csvs = []
        for f in tqdm(metas_df.index, total=len(metas_df.index), desc="Loading simulation data"):
            csvs.append(pd.read_csv((BASE_PATH / f).with_suffix(".csv")))

        sims_df = pd.concat(csvs)
    else:
        raise ValueError(f"Unknown file format for {metas_df.index[0]}")
    return csvs, f, sims_df


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
        r"""
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
    grouped[validity_columns].all()[:100]
    return


@app.cell
def _(metas_df, pd, validity_columns):
    def process_sim(group):
        sim_file = group.iloc[0]["sim_file"]
        fault_start_idx = metas_df.loc[sim_file, "fault_start_time"]-1
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
    processed[:100]
    return (processed,)


@app.cell
def _(metas_df, pd, processed):
    # Evaluate metrics
    results = pd.concat([processed, metas_df], axis="columns")
    results[:100]
    return (results,)


@app.cell
def _(results):
    # Save results to file
    results.to_csv("results.csv")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Below is a plot the effect of eps_scaler on fault detection (TP+FP w.r.t. eps_scaler)
        The high TP+FP when eps_scaler is small is likely due to the high number of false positives (modeling error falsely detected as fault).
        """
    )
    return


@app.cell
def _(plt, results_with_metrics, sns):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=results_with_metrics, x="eps_scaler", hue="is_fault_detected", multiple="stack", bins=50)
    plt.xlabel("eps_scaler")
    plt.ylabel("Count")
    plt.title("Effect of eps_scaler on fault detection")
    plt.show()
    return


@app.cell
def _(np, plt, results_with_metrics, sns):
    def _():
        # Assume results_with_metrics DataFrame with 'eps_scaler' and 'precision'
        num_bins = 50
        bins = np.linspace(results_with_metrics['eps_scaler'].min(),
                           results_with_metrics['eps_scaler'].max(),
                           num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Compute absolute counts per bin
        counts, _ = np.histogram(results_with_metrics['eps_scaler'], bins=bins)

        # Create stacked subplots with shared x-axis
        fig, (ax_top, ax_bottom) = plt.subplots(
            2, 1, 
            figsize=(10, 7), 
            sharex=True, 
            gridspec_kw={'height_ratios': [1, 4], 'hspace': 0.05}
        )

        # Top: Absolute count bar plot
        ax_top.bar(bin_centers, counts, width=(bins[1] - bins[0]), color='gray')
        ax_top.set_ylabel("Count")
        ax_top.set_yticks([0, max(counts)])  # Show just 0 and max for minimalism
        ax_top.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        sns.despine(ax=ax_top, bottom=True)

        # Bottom: Normalized proportion histogram (your main plot)
        sns.histplot(
            data=results_with_metrics,
            x="eps_scaler",
            hue="precision",
            multiple="fill",
            bins=bins,
            ax=ax_bottom
        )
        ax_bottom.set_ylabel("Proportion")
        ax_bottom.set_xlabel("eps_scaler")
        ax_bottom.set_title("Effect of eps_scaler on precision")
        sns.despine(ax=ax_bottom)

        plt.tight_layout()
        return plt.show()


    _()
    return


@app.cell
def avg_precision_and_recall(np, pd, plt, results_with_metrics):
    # Graph of average precision and recall

    def _():
        overall_avg_precision = results_with_metrics['precision'].mean()
        overall_avg_recall = results_with_metrics['recall'].mean()

        print(f"Overall average precision: {overall_avg_precision}")
        print(f"Overall average recall: {overall_avg_recall}")

        if results_with_metrics['eps_scaler'].min() == results_with_metrics['eps_scaler'].max():
            print(f"There's only 1 unique eps_scaler value: {results_with_metrics['eps_scaler'].min()}. Skipping graph.")

            return
        num_bins = 50
        bins = np.linspace(results_with_metrics['eps_scaler'].min(),
                           results_with_metrics['eps_scaler'].max(),
                           num_bins + 1)
        counts, bin_edges = np.histogram(results_with_metrics['eps_scaler'], bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Calculate average precision
        avg_precision = results_with_metrics.groupby(pd.cut(results_with_metrics['eps_scaler'], bins), observed=True)['precision'].mean()
        avg_recall = results_with_metrics.groupby(pd.cut(results_with_metrics['eps_scaler'], bins), observed=True)['recall'].mean()

        fig, ax1 = plt.subplots(figsize=(10, 6))
        plot_avg_precision = ax1.plot(bin_centers, avg_precision, marker='o', label='Average Precision')
        plot_avg_recall = ax1.plot(bin_centers, avg_recall, marker='o', label='Average Recall')
        ax1.set_xlabel("eps_scaler")
        ax1.set_ylabel("Precision / Recall")
        ax1.set_title("Average Precision and Recall vs eps_scaler")
        ax1.grid()
        ax1.set_ylim([0,1])

        ax2 = ax1.twinx()
        plot_num_sims = ax2.bar(bin_centers, counts, width=(bin_edges[1] - bin_edges[0]), 
                alpha=0.2, color='gray', label='# of simulations', zorder=0)
        ax2.set_ylabel("# of simulations")
        ax2.set_ylim(bottom=0)  # always start count axis at 0

        # Step 5: Optional: adjust legends for clarity
        plots = plot_avg_precision + plot_avg_recall + [plot_num_sims]
        labels = [l.get_label() for l in plots]
        ax2.legend(plots, labels, loc="upper right")

        plt.tight_layout()

        plt.show()
    _()
    return


@app.cell
def _(np, plt, results_with_metrics, sns):
    def _():
        # Your data: results_with_metrics (should be a DataFrame)
        # Assume 'eps_scaler' is continuous, 'precision' is categorical/discrete

        # Step 1: Prepare bins
        num_bins = 50
        bins = np.linspace(results_with_metrics['eps_scaler'].min(),
                           results_with_metrics['eps_scaler'].max(),
                           num_bins + 1)

        # Step 2: Calculate bin centers and absolute counts
        counts, bin_edges = np.histogram(results_with_metrics['eps_scaler'], bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Step 3: Plot the proportion histogram
        fig, ax1 = plt.subplots(figsize=(10, 6))
        sns.histplot(
            data=results_with_metrics,
            x="eps_scaler",
            hue="precision",
            multiple="fill",
            bins=bins,
            ax=ax1,
            legend=True
        )
        ax1.set_ylabel("Proportion")
        ax1.set_xlabel("eps_scaler")
        ax1.set_title("Effect of eps_scaler on precision (with absolute counts)")

        # Step 4: Plot absolute counts using a secondary y-axis
        ax2 = ax1.twinx()
        ax2.bar(bin_centers, counts, width=(bin_edges[1] - bin_edges[0]), 
                alpha=0.2, color='gray', label='Absolute count', zorder=0)
        ax2.set_ylabel("Absolute Count")
        ax2.set_ylim(bottom=0)  # always start count axis at 0

        # Step 5: Optional: adjust legends for clarity
        ax2.legend(loc="upper right")

        plt.tight_layout()
        plt.show()

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
        sns.histplot(data=results_with_metrics, x="eps_scaler", hue="precision", multiple="fill", bins=50)
        plt.xlabel("eps_scaler")
        plt.ylabel("Proportion")
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
        sns.histplot(data=results_with_metrics, x="eps_scaler", hue="recall", multiple="fill", bins=50)
        plt.xlabel("eps_scaler")
        plt.ylabel("Proportion")
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
def _(fault_sensor_dropdown, plt, results_with_metrics, sns):
    def plot_fault_bias_effects(results_with_metrics, fault_sensor):
        filtered_results = results_with_metrics[
            (results_with_metrics["num_faulty_sensors"] == 1) &
            (results_with_metrics["fault_0_type"] == "bias") &
            (results_with_metrics['fault_0_sensor'] == fault_sensor)
        ]

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(ncols=2, figsize=(10, 3), sharex=True)

        # Precision plot
        sns.histplot(data=filtered_results, x="fault_0_bias", hue="precision", multiple="stack", bins=50, ax=axes[0])
        axes[0].set_ylabel("Count")
        axes[0].set_title("Effect of Fault Bias on Precision (Bias faults)")

        # Recall plot
        sns.histplot(data=filtered_results, x="fault_0_bias", hue="recall", multiple="stack", bins=50, ax=axes[1])
        axes[1].set_xlabel("Fault Bias")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Effect of Fault Bias on Recall (Bias faults)")

        plt.tight_layout()
        return fig

    # Example usage
    plot_fault_bias_effects(results_with_metrics, fault_sensor_dropdown.value)
    return (plot_fault_bias_effects,)


@app.cell
def _(fault_sensor_dropdown, plt, results_with_metrics, sns):
    def _(results_with_metrics, fault_sensor):
        filtered_results = results_with_metrics[
            (results_with_metrics["num_faulty_sensors"] == 1) &
            (results_with_metrics["fault_0_type"] == "drift") &
            (results_with_metrics['fault_0_sensor'] == fault_sensor)
        ]

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(ncols=2, figsize=(10, 3), sharex=True)

        # Precision plot
        sns.histplot(data=filtered_results, x="fault_0_drift_rate", hue="precision", multiple="stack", bins=50, ax=axes[0])
        axes[1].set_xlabel("Fault Drift Rate")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Effect of Fault Drift Rate on Precision (Drift faults)")

        # Recall plot
        sns.histplot(data=filtered_results, x="fault_0_drift_rate", hue="recall", multiple="stack", bins=50, ax=axes[1])
        axes[1].set_xlabel("Fault Drift Rate")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Effect of Fault Drift Rate on Recall (Drift faults)")

        plt.tight_layout()
        return fig

    # Example usage
    _(results_with_metrics, fault_sensor_dropdown.value)
    return


@app.cell
def _(fault_sensor_dropdown, plt, results_with_metrics, sns):
    def _(results_with_metrics, fault_sensor):
        filtered_results = results_with_metrics[
            (results_with_metrics["num_faulty_sensors"] == 1) &
            (results_with_metrics["fault_0_type"] == "noise") &
            (results_with_metrics['fault_0_sensor'] == fault_sensor)
        ]

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(ncols=2, figsize=(10, 3), sharex=True)

        # Precision plot
        sns.histplot(data=filtered_results, x="fault_0_amplitude", hue="precision", multiple="stack", bins=50, ax=axes[0])
        axes[1].set_xlabel("Fault Amplitude")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Effect of Fault Amplitude on Precision (Noise faults)")

        # Recall plot
        sns.histplot(data=filtered_results, x="fault_0_amplitude", hue="recall", multiple="stack", bins=50, ax=axes[1])
        axes[1].set_xlabel("Fault Amplitude")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Effect of Fault Amplitude on Recall (Noise faults)")

        plt.tight_layout()
        return fig

    # Example usage
    _(results_with_metrics, fault_sensor_dropdown.value)
    return


@app.cell
def _(fault_sensor_dropdown, plt, results_with_metrics, sns):
    def _(results_with_metrics, fault_sensor):
        filtered_results = results_with_metrics[
            (results_with_metrics["num_faulty_sensors"] == 1) &
            (results_with_metrics["fault_0_type"] == "spike") &
            (results_with_metrics['fault_0_sensor'] == fault_sensor)
        ]

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(ncols=2, figsize=(10, 3), sharex=True)

        # Precision plot
        sns.histplot(data=filtered_results, x="fault_0_amplitude", hue="precision", multiple="stack", bins=50, ax=axes[0])
        axes[1].set_xlabel("Fault Amplitude")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Effect of Fault Amplitude on Precision (Spike Faults)")

        # Recall plot
        sns.histplot(data=filtered_results, x="fault_0_amplitude", hue="recall", multiple="stack", bins=50, ax=axes[1])
        axes[1].set_xlabel("Fault Amplitude")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Effect of Fault Amplitude on Recall (Spike Faults)")

        plt.tight_layout()
        return fig

    # Example usage
    _(results_with_metrics, fault_sensor_dropdown.value)
    return


@app.cell
def _(fault_sensor_dropdown, results_with_metrics):
    results_with_metrics[(results_with_metrics["num_faulty_sensors"] == 1) & (results_with_metrics["fault_0_type"] == "bias") & (results_with_metrics['fault_0_sensor'] == fault_sensor_dropdown.value)][:100]
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
def _(EXP_NAME, OUTPUT_DIR, results_with_metrics):
    # save results_with_metrics to a file
    results_with_metrics.to_parquet(OUTPUT_DIR / f"results_with_metrics_{EXP_NAME}.parquet")
    return


@app.cell
def _(results_with_metrics):
    results_with_metrics[:100]
    return


@app.cell
def _(results_with_metrics):
    results_with_metrics["detection_delay"] = results_with_metrics["time_of_detection"] - results_with_metrics["fault_start_time_s"]
    return


@app.cell
def _(results_with_metrics):
    results_with_metrics.groupby(["fault_0_type","fault_1_type"],dropna=False)["precision"].describe()
    return


@app.cell
def _(results_with_metrics):
    results_with_metrics[(results_with_metrics["eps_scaler"]>0.8) & (results_with_metrics["eps_scaler"]<1.2) ].groupby(["fault_0_type","fault_1_type"],dropna=False)["precision"].describe()
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


@app.cell
def _(results_with_metrics):
    results_single_sensor_faults = results_with_metrics[results_with_metrics["num_faulty_sensors"] == 1]
    results_single_sensor_faults.groupby(["fault_0_sensor", "fault_0_type"])["precision"].describe()
    return (results_single_sensor_faults,)


@app.cell
def _(results_with_metrics):
    # Single sensor precision recall table
    def _():
        single_sensor_faults = results_with_metrics[results_with_metrics["num_faulty_sensors"] == 1]

        # Format the table to combine mean and std into a single column with "mean ± std" format
        formatted_table = single_sensor_faults.groupby(["fault_0_type", "fault_0_sensor"])\
            [["precision", "recall"]].agg(["mean", "std"])

        # Combine mean and std into a single string
        for metric in ["precision", "recall"]:
            formatted_table[(metric, "combined")] = formatted_table.apply(
                # lambda row: f"{row[(metric, 'mean')]:.2f} ± {row[(metric, 'std')]:.2f}", axis=1
                lambda row: row[(metric, 'mean')], axis=1
            )

        # Keep only the combined column
        formatted_table = formatted_table[[("precision", "combined"), ("recall", "combined")]]

        # Rename columns for clarity
        formatted_table.columns = ["Precision", "Recall"]

        # Compute mean and std for detection delay per fault type and sensor
        detection_delay_stats = single_sensor_faults.groupby(["fault_0_type", "fault_0_sensor"])\
            ["detection_delay"].agg(["mean", "std"])

        # Format detection delay as "mean ± std"
        detection_delay_stats["Detection Delay (s)"] = detection_delay_stats.apply(
            # lambda row: f"{row['mean']:.2f} ± {row['std']:.2f}", axis=1
            lambda row: row['mean'], axis=1
        )

        # Keep only the formatted column
        detection_delay_stats = detection_delay_stats[["Detection Delay (s)"]]

        # Merge with the existing table
        formatted_table = formatted_table.join(detection_delay_stats)

        # # Compute the sample count for each fault type and sensor combination
        # sample_counts = single_sensor_faults.groupby(["fault_0_type", "fault_0_sensor"]).size()

        # # Convert to DataFrame and rename column
        # sample_counts = sample_counts.to_frame(name="Sample Count")

        # # Merge with the formatted table
        # formatted_table = formatted_table.join(sample_counts)

        # rename index for readability
        formatted_table.index.names = ["Type", "Sensor"]

        # Use 1-indexed sensors instead of 0-indexed sensors
        formatted_table.index = formatted_table.index.set_levels(formatted_table.index.levels[1] + 1, level=1)

        print(formatted_table.to_latex(
            column_format="c" * (len(formatted_table.columns) + len(formatted_table.index.names)),
            multicolumn_format="c",
            float_format="{:0.2f}".format,
        ))

        return formatted_table
    _()
    return


@app.cell
def _(results_with_metrics):
    # Single sensor precision recall table (det rate)
    def _():
        single_sensor_faults = results_with_metrics[results_with_metrics["num_faulty_sensors"] == 1]

        # Format the table to combine mean and std into a single column with "mean ± std" format
        formatted_table = single_sensor_faults.groupby(["fault_0_type", "fault_0_sensor"])\
            [["precision"]].agg(["mean", "std"])

        # Combine mean and std into a single string
        for metric in ["precision"]:
            formatted_table[(metric, "combined")] = formatted_table.apply(
                lambda row: f"{row[(metric, 'mean')]:.2f} ± {row[(metric, 'std')]:.2f}", axis=1
                # lambda row: row[(metric, 'mean')], axis=1
            )

        # Keep only the combined column
        formatted_table = formatted_table[[("precision", "combined")]]

        # Rename columns for clarity
        formatted_table.columns = ["Det. Rate"]

        # Compute mean and std for detection delay per fault type and sensor
        detection_delay_stats = single_sensor_faults.groupby(["fault_0_type", "fault_0_sensor"])\
            ["detection_delay"].agg(["mean", "std"])

        # Format detection delay as "mean ± std"
        detection_delay_stats["Det. Delay (s)"] = detection_delay_stats.apply(
            lambda row: f"{row['mean']:.2f} ± {row['std']:.2f}", axis=1
            # lambda row: row['mean'], axis=1
        )

        # Keep only the formatted column
        detection_delay_stats = detection_delay_stats[["Det. Delay (s)"]]

        # Merge with the existing table
        formatted_table = formatted_table.join(detection_delay_stats)

        # # Compute the sample count for each fault type and sensor combination
        # sample_counts = single_sensor_faults.groupby(["fault_0_type", "fault_0_sensor"]).size()

        # # Convert to DataFrame and rename column
        # sample_counts = sample_counts.to_frame(name="Sample Count")

        # # Merge with the formatted table
        # formatted_table = formatted_table.join(sample_counts)

        # rename index for readability
        formatted_table.index.names = ["Type", "Sensor"]

        # Use 1-indexed sensors instead of 0-indexed sensors
        formatted_table.index = formatted_table.index.set_levels(formatted_table.index.levels[1] + 1, level=1)

        print(formatted_table.to_latex(
            column_format="c" * (len(formatted_table.columns) + len(formatted_table.index.names)),
            multicolumn_format="c",
            float_format="{:0.2f}".format,
        ))

        return formatted_table
    _()
    return


@app.cell
def _(HTML, display, pd, results_with_metrics):
    # Threshold-based metrics
    def _():
        single_sensor_faults = results_with_metrics[results_with_metrics["num_faulty_sensors"] == 1]

        THRESHOLD_COLUMN_NAMES = {
            'bias': 'fault_0_bias',
            'drift': 'fault_0_drift_rate',
            'noise': 'fault_0_amplitude',
            'spike': 'fault_0_amplitude',
        }
        THRESHOLDS = {
            'bias': [0,0,0.9,3.5,3,0.7],
            'drift': [0,0,0.8,0.8,0,0.1],
            'noise': [0,0,1.5,1.5,1,0.31],
            'spike': [0,0,0.8,3,3,0.65],
        }

        def compute_threshold_stats(group, threshold, threshold_column):
            below_thresh = group[group[threshold_column].abs() < threshold]
            above_thresh = group[group[threshold_column].abs() >= threshold]

            stats = {}
            stats[("Thresh", "")] = threshold  # Keep this separate from Below/Above sections

            for subset, label in zip([below_thresh, above_thresh], ["Below Threshold", "Above Threshold"]):
                # stats[(label, "Samples")] = len(subset)
                if not subset.empty:
                    # stats[(label, "Det Rate")] = subset['precision'].mean()
                    stats[(label, r"\#")] = len(subset)
                    stats[(label, "Det Rate")] = f"{subset['precision'].mean():.2f} ± {subset['precision'].std():.2f}"
                    if len(subset[subset['precision'] > 0]) > 0:
                        # stats[(label, "Det Delay")] = subset[subset['precision'] > 0]['detection_delay'].mean()
                        stats[(label, "Det Delay")] = f"{subset[subset['precision'] > 0]['detection_delay'].mean():.2f} ± {subset[subset['precision'] > 0]['detection_delay'].std(ddof=0):.2f}"
                    else:
                        stats[(label, "Det Delay")] = "N/A"
                else:
                    stats[(label, r"\#")] = 0
                    stats[(label, "Det Rate")] = "N/A"
                    stats[(label, "Det Delay")] = "N/A"

            return pd.Series(stats)

        threshold_based_stats = single_sensor_faults.groupby(["fault_0_type", "fault_0_sensor"]).apply(
            lambda group: compute_threshold_stats(
                group,
                THRESHOLDS[group["fault_0_type"].iloc[0]][group["fault_0_sensor"].iloc[0]],
                THRESHOLD_COLUMN_NAMES[group["fault_0_type"].iloc[0]],
            )
        )

        # Ensure column names are tuples
        threshold_based_stats.columns = pd.MultiIndex.from_tuples(threshold_based_stats.columns)

        # Rename index for readability
        threshold_based_stats.index.names = ["Type", "Sen"]

        # Use 1-indexed sensors instead of 0-indexed sensors
        threshold_based_stats.index = threshold_based_stats.index.set_levels(threshold_based_stats.index.levels[1] + 1, level=1)

        # Print as LaTeX table for reference
        print(threshold_based_stats.to_latex(
            column_format="c" * (len(threshold_based_stats.columns) + len(threshold_based_stats.index.names)),
            multicolumn_format="c",
            float_format="{:0.2f}".format,
        ))

        display(HTML(threshold_based_stats.to_html()))

    _()
    return


@app.cell
def _(HTML, display, pd, results_with_metrics):
    # Threshold-based metrics (condensed)
    def _():
        single_sensor_faults = results_with_metrics[results_with_metrics["num_faulty_sensors"] == 1]

        THRESHOLD_COLUMN_NAMES = {
            'bias': 'fault_0_bias',
            'drift': 'fault_0_drift_rate',
            'noise': 'fault_0_amplitude',
            'spike': 'fault_0_amplitude',
        }
        THRESHOLDS = {
            'bias': [0,0,0.9,3.5,3,0.7],
            'drift': [0,0,0.8,0.8,0,0.15],
            'noise': [0,0,1.5,1.5,1,0.31],
            'spike': [0,0,0.8,3,3,0.65],
        }

        def compute_threshold_stats(group, threshold, threshold_column):
            below_thresh = group[group[threshold_column].abs() < threshold]
            above_thresh = group[group[threshold_column].abs() >= threshold]

            stats = {}
            stats[("Threshold", "")] = threshold  # Keep this separate from Below/Above sections

            for subset, label in zip([below_thresh, above_thresh], ["Below Threshold", "Above Threshold"]):
                # stats[(label, "Samples")] = len(subset)
                if not subset.empty:
                    stats[(label, "Rate")] = subset['precision'].mean()
                    # stats[(label, "Rate")] = f"{subset['precision'].mean():.2f} ± {subset['precision'].std():.2f}"
                    if len(subset[subset['precision'] > 0]) > 0:
                        stats[(label, "Delay")] = subset[subset['precision'] > 0]['detection_delay'].mean()
                        # stats[(label, "Delay")] = f"{subset[subset['precision'] > 0]['detection_delay'].mean():.2f} ± {subset[subset['precision'] > 0]['detection_delay'].std(ddof=0):.2f}"
                    else:
                        stats[(label, "Delay")] = "N/A"
                else:
                    stats[(label, "Rate")] = "N/A"
                    stats[(label, "Delay")] = "N/A"

            return pd.Series(stats)

        threshold_based_stats = single_sensor_faults.groupby(["fault_0_type", "fault_0_sensor"]).apply(
            lambda group: compute_threshold_stats(
                group,
                THRESHOLDS[group["fault_0_type"].iloc[0]][group["fault_0_sensor"].iloc[0]],
                THRESHOLD_COLUMN_NAMES[group["fault_0_type"].iloc[0]],
            )
        )

        # Ensure column names are tuples
        threshold_based_stats.columns = pd.MultiIndex.from_tuples(threshold_based_stats.columns)

        # Rename index for readability
        threshold_based_stats.index.names = ["Type", "Sensor"]

        # Use 1-indexed sensors instead of 0-indexed sensors
        threshold_based_stats.index = threshold_based_stats.index.set_levels(threshold_based_stats.index.levels[1] + 1, level=1)

        # Print as LaTeX table for reference
        print(threshold_based_stats.to_latex(
            column_format="c" * (len(threshold_based_stats.columns) + len(threshold_based_stats.index.names)),
            multicolumn_format="c",
            float_format="{:0.2f}".format,
        ))

        display(HTML(threshold_based_stats.to_html()))

    _()
    return


@app.cell
def _(results_with_metrics):
    # Multi-sensor precision and recall table

    # Filter data for cases where exactly two sensors are faulty
    two_sensor_faults = results_with_metrics[results_with_metrics["num_faulty_sensors"] == 2]

    # Compute mean and std for precision and recall grouped by sensor pairs
    sensor_pair_table = two_sensor_faults.groupby(["fault_0_sensor", "fault_1_sensor"])[["precision", "recall"]].agg(["mean", "std"])

    # Format columns to display mean ± std
    for metric in ["precision", "recall"]:
        sensor_pair_table[(metric, "combined")] = sensor_pair_table.apply(
            lambda row: f"{row[(metric, 'mean')]:.2f} ± {row[(metric, 'std')]:.2f}", axis=1
        )

    # Keep only the combined columns
    sensor_pair_table = sensor_pair_table[[("precision", "combined"), ("recall", "combined")]]
    sensor_pair_table.columns = ["Precision (Mean ± Std)", "Recall (Mean ± Std)"]

    # Convert to square matrix format for compact display
    precision_pivot = sensor_pair_table["Precision (Mean ± Std)"].unstack().fillna("-")
    precision_pivot.columns = precision_pivot.columns.astype(int) + 1
    precision_pivot.index = precision_pivot.index.astype(int) + 1
    precision_pivot.index.names = ["Sensor 0"]
    precision_pivot.columns.names = ["Sensor 1"]

    recall_pivot = sensor_pair_table["Recall (Mean ± Std)"].unstack().fillna("-")
    recall_pivot.columns = recall_pivot.columns.astype(int) + 1
    recall_pivot.index = recall_pivot.index.astype(int) + 1
    recall_pivot.index.names = ["Sensor 0"]
    recall_pivot.columns.names = ["Sensor 1"]

    # Display the results
    print("\nPrecision (Mean ± Std) by Sensor Pair:")
    print(precision_pivot)
    print(precision_pivot.to_latex(column_format="c"*(len(precision_pivot.columns)+1)))

    print("\nRecall (Mean ± Std) by Sensor Pair:")
    print(recall_pivot)
    print(recall_pivot.to_latex(column_format="c"*(len(recall_pivot.columns)+1)))
    return (
        metric,
        precision_pivot,
        recall_pivot,
        sensor_pair_table,
        two_sensor_faults,
    )


if __name__ == "__main__":
    app.run()
