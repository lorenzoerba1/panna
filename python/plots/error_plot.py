import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
from pathlib import Path

if __name__ == "__main__":
    mpl.use("WebAgg")
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{siunitx} \usepackage{sansmath} \sansmath",
        }
    )
    
    filepath_prefix = Path(__file__).resolve().parents[2]
    filepath = os.path.join(filepath_prefix, "results/weight_results.csv")
    df = pd.read_csv(filepath)
    sns.set_theme(palette="muted", style="white", context="paper")

    df["Dataset"] = df["Dataset"].str.split("-").str[0]
    df["Dataset"] = df["Dataset"].str.replace("datasets/", "")
    df["group"] = df.groupby(["Dataset"])["Weight"].transform(lambda x: x.min())
    df["Relative Error"] = (df["Weight"] - df["group"]) / df["group"]
    df = df[df["Algorithm"] != "Exact"]
    deltas = sorted(df["delta"].unique())

    fig, axs = plt.subplots(figsize=(6, 3), ncols=len(deltas)+1, sharey=True)
    palette = sns.color_palette("muted")
    algorithms = df["Algorithm"].unique()
    color_map = dict(zip(algorithms, palette))

    for i, delta in enumerate(deltas):
        relative_data = df[df["delta"] == delta]
        bplot = sns.boxplot( 
            data=relative_data,
            x="Dataset",
            y="Relative Error",
            hue="Algorithm",
            palette=color_map,
            fill=False,
            boxprops=dict(alpha=0.7),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=2),
            medianprops=dict(linewidth=2),
            showfliers=False,
            ax=axs[i],
            legend=True if i == 0 else False,
        )        
        sns.despine()
        axs[i].spines["left"].set_bounds(df["Relative Error"].min(), df["Relative Error"].max())

        # Change the names of the legend
    new_names = [
            r"K+",
            r"K+ $\epsilon$ 0.2",
            r"K+ $\epsilon$ 0.5",
            r"K+ $\epsilon$ 1.0",
            r"K+ $\epsilon$ 5.0",
            r"K+ $\epsilon$ 10.0",
            r"K+ $\epsilon$ 20.0",
        ]
    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].legend_.remove()
    label_map = dict(zip(labels, new_names[:len(labels)]))
    handles = [h for h, l in zip(handles, labels) if l in label_map]
    labels = [label_map[l] for l in labels if l in label_map]

    axs[-1].axis("off")
    axs[-1].legend(handles, labels, loc="center", frameon=False, title="Algorithm")

        
    plt.show()
