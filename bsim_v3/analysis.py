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
    BASE_PATH = Path("exp/bsim_v3/bias-sweep-5")
    return (BASE_PATH,)


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
def _(pd, processed_metas):
    metas_df = pd.DataFrame(processed_metas).set_index("sim_file")
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
def _(pd, validity_columns):
    def process_sim(group):
        exists_invalid_sensors = (group[validity_columns] == False).any(axis=1) # whether each time step (row) has invalid sensors
        if exists_invalid_sensors.any():
            first_invalid_index = exists_invalid_sensors.idxmax()  # Get index of first invalid occurrence
            first_invalid_row = group.loc[first_invalid_index]

            ret = pd.Series({
                "time_of_detection": first_invalid_row["t"],
                "is_fault_detected": True,
                **first_invalid_row[validity_columns].to_dict(),
            })
        else:
            ret = pd.Series({
                "time_of_detection": None,
                "is_fault_detected": False,
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
    # Plot the effect of eps_scaler on fault detection
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=results, x="eps_scaler", hue="is_fault_detected", multiple="stack", bins=50)
    plt.xlabel("eps_scaler")
    plt.ylabel("Count")
    plt.title("Effect of eps_scaler on fault detection")
    plt.show()
    return plt, sns


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


if __name__ == "__main__":
    app.run()
