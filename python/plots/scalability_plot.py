import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
from pathlib import Path

if __name__ == "__main__":
    mpl.use("WebAgg")
    # Read the CSV file
    filepath_prefix = Path(__file__).resolve().parents[2]
    filepath = os.path.join(filepath_prefix, "results/dimensionality_scalability.csv")
    df = pd.read_csv(filepath)
    sns.set_theme(palette="muted", style="white")
    

    # Create a line plot for the scalability of HDBSCAN
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df, x="D", y="Time (s)", hue="Algorithm", marker="o", ax=ax)

    # Set the title and labels
    ax.set_title("Dimensionality Scalability")
    ax.set_xlabel("Number of Dimensions (D)")
    ax.set_ylabel("Time (s)")
    ax.set_yscale("log")  # Use logarithmic scale for better visibility of time differences
    ax.set_xscale("log")  # Use logarithmic scale for dimensions
    sns.despine()
    # Apply Tufte style, axis lines only between the lowest and highest values Time (s) where algorithm is K+ or K+ ɛ 5.0
    ax.spines["left"].set_bounds(df[df["Algorithm"].isin(["K+", "K+ ɛ 5.0"])]["Time (s)"].min(), df[df["Algorithm"].isin(["K+", "K+ ɛ 5.0"])]["Time (s)"].max())
    ax.spines["bottom"].set_bounds(df["D"].min(), df["D"].max())
    # Add a grid and legend
    ax.legend()
    # Save the plot as a PNG file
    plt.show()
    plt.savefig("results/dimensionality_scalability_plot.png", dpi=300)

    # Open the length scalability CSV file
    filepath = os.path.join(filepath_prefix, "results/length_scalability.csv")
    df_length = pd.read_csv(filepath)
    # Get the unique dimensions
    dimensions = sorted(df_length["D"].unique())
    # Create a line plot for the scalability, to avoid a spaghetti plot, let's use a plot for each dimension of the data
    fig, axs = plt.subplots(
        figsize=(10, 6),
        nrows=len(dimensions)//2,
        ncols=2,
        constrained_layout=True,
        # sharey=True,
        # sharex=True,
    )
    colors = dict(zip(df_length["Algorithm"].unique(), sns.color_palette(n_colors=len(df_length["Algorithm"].unique()))))
    for i, dim in enumerate(dimensions):
        subset = df_length[df_length["D"] == dim]
        sns.lineplot(
            data=subset, x="n", y="Time (s)", hue="Algorithm", marker="o", ax=axs[i//2, i%2], errorbar=None, palette=colors, legend=True if i == 0 else False
        )
        # Deactivate the name of the axis
        axs[i//2, i%2].set_ylabel("")
        axs[i//2, i%2].set_xlabel("")
        axs[i//2, i%2].set_title(f"Dimension {dim}")
        axs[i//2, i%2].set_xscale("log")
        axs[i//2, i%2].set_yscale("log")
        sns.despine()
        axs[i//2, i%2].spines["left"].set_bounds(subset[subset["Algorithm"].isin(["K+", "K+ ɛ 5.0"])]["Time (s)"].min(), subset[subset["Algorithm"].isin(["K+", "K+ ɛ 5.0"])]["Time (s)"].max())
        axs[i//2, i%2].spines["bottom"].set_bounds(subset["n"].min(), subset["n"].max())
        if i//2 == len(dimensions)//2 - 1:
            axs[i//2, i%2].set_xlabel("Number of Samples (n)")
        if i%2 == 0:
            axs[i//2, i%2].set_ylabel("Time (s)")
    # Set common labels
    plt.ylabel("Time (s)")
    plt.xlabel("Number of Samples (n)")
    plt.suptitle("Length Scalability by Dimension")
    # Add a single legend for all subplots
    # Set the title and labels
    plt.show()
    plt.savefig("results/length_scalability_plot.png", dpi=600)
