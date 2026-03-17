# /// script
# dependencies = [
#     "great-tables==0.21.0",
#     "marimo",
#     "numpy==2.4.3",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import polars.selectors as cs
    import great_tables
    from great_tables import GT

    return cs, pl


@app.cell
def _(pl):
    pkey = [
        "algorithm",
        "parameters",
        "machine",
        "dataset",
        "dataset_sample_frac",
        "dataset_sample_seed"
    ]
    all_results = (
        pl.read_ndjson("emst.json")
        .filter(pl.col("version") == "1")
        .filter(pl.col("timestamp") == pl.col("timestamp").max().over(pkey))
    )
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


@app.cell
def _(all_results, cs, pl):
    approximate_full = (
        all_results
        .filter(pl.col("dataset_sample_frac").is_null())
        .filter(pl.col("parameters").struct.field("repetitions") == 1024)
        .with_columns(
            ground_weight = pl.col("emst_weight").min().over("dataset")
        )
        .with_columns(
            weight_factor= pl.col("emst_weight") / pl.col("emst_weight").min().over("dataset"),
            relative_error = (pl.col("emst_weight") - pl.col("ground_weight")) / pl.col("ground_weight")
        )
        .filter(pl.col("parameters").struct.field("epsilon") >= 0.0)
        .select(pl.col("dataset").str.replace("-[0-9]+-(euclidean|angular|normalized)", ""), 
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

    approximate_full_tbl = (
        approximate_full
        .pivot(index=["dataset"], on="epsilon", values=["running_time_s", "relative_error"])
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


if __name__ == "__main__":
    app.run()
