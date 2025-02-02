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
    mo.md("""## Fault detection performance""")
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
    is_fault_detected = grouped[validity_columns].all().apply(lambda x: not x.all(), axis=1)
    is_fault_detected
    return (is_fault_detected,)


@app.cell
def _(metas_df):
    eps_scalers = metas_df["eps_scaler"]
    eps_scalers
    return (eps_scalers,)


@app.cell
def _(eps_scalers, is_fault_detected, pd):
    # combine derived results
    results = pd.DataFrame({
        "eps_scaler": eps_scalers,
        "is_fault_detected": is_fault_detected,
    })
    results[:100]
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
def _(is_fault_detected, metas_df, pd, plt, sims_df, sns):
    def _():
        filtered_metas = metas_df[(metas_df["num_faulty_sensors"] == 1) & (metas_df["fault_0_type"] == "bias")]
        filtered_sims = sims_df.loc[sims_df.index.isin(filtered_metas.index)]

        results = pd.DataFrame({
            "eps_scaler": filtered_metas["eps_scaler"],
            "is_fault_detected": is_fault_detected,
            "fault_0_bias": filtered_metas["fault_0_bias"]
        })

        # import matplotlib.pyplot as plt
        # import seaborn as sns
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        sns.histplot(data=results, x="fault_0_bias", hue="is_fault_detected", multiple="stack", bins=50)
        plt.xlabel("Fault Bias")
        plt.ylabel("Count")
        plt.title("Effect of Fault Bias on Detection")
        return plt.gca()


    _()
    return


if __name__ == "__main__":
    app.run()
