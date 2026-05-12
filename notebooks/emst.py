# /// script
# dependencies = [
#     "altair==6.0.0",
#     "great-tables==0.21.0",
#     "marimo",
#     "numpy==2.4.4",
#     "polars==1.40.1",
#     "seaborn==0.13.2",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    import polars.selectors as cs
    import great_tables
    from great_tables import GT
    import seaborn as sns

    return cs, mo, pl, sns


@app.cell
def _(mo):
    sel_algo_version = mo.ui.dropdown(
            ["0", "0.2.2", "1", "2", "3", "4"],
            label="algorithm version",
            value="4",
        )
    sel_algo_version
    return (sel_algo_version,)


@app.cell
def _(mo, pl, sel_algo_version):
    pkey = [
        "algorithm",
        "parameters",
        "machine",
        "dataset",
        "dataset_sample_frac",
        "dataset_sample_seed"
    ]
    full_results = (
        pl.read_ndjson("results/emst.json", infer_schema_length=None)
        .filter(pl.col("version") == sel_algo_version.value)
        .filter(pl.col("timestamp") == pl.col("timestamp").max().over(pkey))
        .with_columns(pl.col("dataset").str.replace("-[0-9]+-(euclidean|angular|normalized)", ""))
    )
    algo_name = mo.ui.dropdown.from_series(full_results["algorithm"])
    algo_name
    return algo_name, full_results


@app.cell
def _(algo_name, full_results, pl):
    all_results = full_results.filter(pl.col("algorithm") == algo_name.value)
    all_results
    return (all_results,)


@app.cell
def _(all_results, pl):
    exact_full = (
        all_results
        .filter(pl.col("parameters").struct.field("epsilon") == 0.0)
        .filter(pl.col("dataset_sample_frac").is_null())
        .select("dataset", 
                "algorithm", 
                pl.col("parameters")
                    .struct.field("repetitions").alias("repetitions"), 
                pl.col("running_time_s")
                    .round(2)
        )
        .sort("dataset", "algorithm", "repetitions")
    )
    exact_full.style
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dataset statistics
    """)
    return


@app.cell
def _(pl):
    sizes = pl.DataFrame([
        {"dataset": "fashion-mnist", "n": 60000, "d": 784},
        {"dataset": "gist", "n": 1000000, "d": 960},
        {"dataset": "glove", "n": 1183514, "d": 100},
        {"dataset": "imagenet", "n": 1281167, "d": 512},
        {"dataset": "landmark-nomic", "n": 760757, "d": 768},
        {"dataset": "nytimes", "n": 290000, "d": 256},
        {"dataset": "sift", "n": 1000000, "d": 128},
        {"dataset": "simplewiki-openai", "n": 260372, "d": 3072},
        {"dataset": "ht", "n": 928991, "d": 11},
        {"dataset": "census", "n": 223223, "d": 500},
        {"dataset": "pamap2", "n": 2872533, "d": 4},
        {"dataset": "chem", "n": 4208261, "d": 12}
    ])
    return (sizes,)


@app.cell
def _(all_results, cs, pl, sizes):
    dataset_stats = (
        all_results
        .filter(pl.col("machine").struct.field("node_name") != "nixos")
        .filter(pl.col("parameters").struct.field("epsilon") == 0.0)
        .filter(pl.col("dataset_sample_frac").is_null())
        .unnest("detail")
        .filter(pl.col("flexibility@0.0").is_null().not_())
        .join(sizes, on="dataset", how="left")
        .sort(pl.col("d"))
        .with_columns(tree_edges = pl.col("n") - 1)
        .with_columns(cs.starts_with("flexibility") / pl.col("tree_edges"))
    )
    return (dataset_stats,)


@app.cell
def _(dataset_stats, mo):
    tree_dataset = mo.ui.dropdown(dataset_stats["dataset"].to_list(), label="select dataset to show tree")
    tree_dataset
    return (tree_dataset,)


@app.cell
def _(dataset_stats, pl, sns, tree_dataset):
    tree_info = dataset_stats.filter(pl.col("dataset") == tree_dataset.value).select("tree_path").to_dicts()[0]
    sns.ecdfplot(
        data=pl.read_parquet(tree_info["tree_path"]),
        x="weight",
    )
    return


@app.cell
def _(cs, dataset_stats):
    dataset_stats_tbl = (
        dataset_stats
        .select("dataset", "n", "d", "mass-frac@0.0", "mass-frac@0.5", "mass-frac@1.0", "flexibility@0.5", "flexibility@1.0")
        .style
        .fmt_percent(columns=cs.starts_with("mass"))
        .fmt_percent(columns=cs.starts_with("flexibility"), decimals=3)
        .fmt_number(columns=["n", "d"], decimals=0)
        .tab_spanner(label="Edge mass (percent)", columns=cs.starts_with("mass"))
        .cols_label_with(fn=lambda c: c.split("@")[-1], columns=cs.starts_with("mass"))
        .tab_spanner(label="Edge flexibility", columns=cs.starts_with("flexibility"))
        .cols_label_with(fn=lambda c: c.split("@")[-1], columns=cs.starts_with("flexibility"))
    )

    with open("notebooks/dataset-stats.tex", "w") as _fp:
        print(dataset_stats_tbl.as_latex(), file=_fp)

    dataset_stats_tbl
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Results
    """)
    return


@app.cell
def _(all_results, pl):
    approximate_full = (
        all_results
        .filter(pl.col("dataset_sample_frac").is_null())
        .filter(pl.col("parameters").struct.field("repetitions") == 512)
        .with_columns(
            ground_weight = pl.col("emst_weight").min().over("dataset")
        )
        .with_columns(
            weight_factor= pl.col("emst_weight") / pl.col("emst_weight").min().over("dataset"),
            relative_error = (pl.col("emst_weight") - pl.col("ground_weight")) / pl.col("ground_weight")
        )
        .filter(pl.col("parameters").struct.field("epsilon") >= 0.0)
        .select(pl.col("dataset"), 
                "algorithm", 
                pl.col("parameters")
                    .struct.field("repetitions").alias("repetitions"), 
                pl.col("parameters")
                    .struct.field("epsilon").alias("epsilon"),
                pl.col("running_time_s"),
                pl.col("relative_error")
        )
        .sort("dataset", "algorithm", "repetitions", "epsilon")
        .filter(pl.col("algorithm") == "k+")
        .select("dataset", "epsilon", "running_time_s", "relative_error")
    )
    approximate_full
    return (approximate_full,)


@app.cell
def _(approximate_full, cs):
    approximate_full_tbl = (
        approximate_full
        .pivot(index=["dataset"], on="epsilon", values=["running_time_s", "relative_error"], aggregate_function="mean")
        .style
        .fmt_number(columns=cs.starts_with("running_time"))
        .fmt_percent(columns=cs.starts_with("relative_error"))
        .tab_spanner(label="Time (s)", columns=cs.starts_with("running_time"))
        .tab_spanner(label="Relative error", columns=cs.starts_with("relative_error"))
        .cols_label_with(fn=lambda c: c.split("_")[-1], columns=cs.starts_with("running_time_s"))
        .cols_label_with(fn=lambda c: c.split("_")[-1], columns=cs.starts_with("relative_error"))
    )

    with open("notebooks/emst-epsilon.tex", "w") as fp:
        print(approximate_full_tbl.as_latex(), file=fp)
    approximate_full_tbl
    return


@app.cell
def _(all_results, mo):
    sel_dataset = mo.ui.dropdown(all_results["dataset"].unique().to_list(), label="dataset", value="glove")
    sel_epsilon = mo.ui.dropdown(all_results["parameters"].struct.field("epsilon").unique().to_list(), label="epsilon", value=0.0)
    mo.vstack([
        sel_dataset,
        sel_epsilon
    ])
    return sel_dataset, sel_epsilon


@app.cell
def _(all_results, pl, sel_dataset, sel_epsilon):
    profile_path = all_results.filter(
        pl.col("dataset") == sel_dataset.value,
        pl.col("parameters").struct.field("epsilon") == sel_epsilon.value,
        pl.col("dataset_sample_frac").is_null()
    ).select("profile_path").to_dicts()[0]["profile_path"]
    profile = pl.read_parquet(profile_path).with_columns(elapsed_s = pl.col("elapsed_ms") / 1000)
    (
        profile.select("elapsed_s", "emst_confirmed_weight", "emst_weight_lower_bound", "emst_total_weight")
            .unpivot(index="elapsed_s", variable_name="type", value_name="weight")
            .plot
            .line(x="elapsed_s", y="weight", color="type")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On `census` there is a peculiar behavior in the plot above, which is probably due to the reindexing of the data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Appendix: utilities
    """)
    return


@app.cell
def _(all_results, pl):
    def download_trees():
        import subprocess as sp
        trees = (
            all_results
            .filter(pl.col("machine").struct.field("node_name") != "nixos")
            .filter(pl.col("parameters").struct.field("epsilon") == 0.0)
            .filter(pl.col("dataset_sample_frac").is_null())
            .unnest("detail")
            .filter(pl.col("flexibility@0.0").is_null().not_())
            .select("tree_path")["tree_path"].to_list()
        )
        for tree in trees:
            cmd = ["rsync", "--progress", "ceccarello@login.dei.unipd.it:/nfsd/lovelace/monaco/" + tree, "results/"]
            sp.run(cmd)

    return


if __name__ == "__main__":
    app.run()
