import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
from pathlib import Path
import h5py
import umap
import umap.plot

if __name__ == "__main__":
    mpl.use("WebAgg")
    sns.set_theme(palette="pastel")
    sns.set_style("dark")
    sample_size = 1500
    filepath_prefix = Path(__file__).resolve().parents[2]
    datasets = [
        "fashion-mnist-784-euclidean.hdf5",
        "glove-100-angular.hdf5",
        "nytimes-256-angular.hdf5",
        "gist-960-euclidean.hdf5",
    ]

    with h5py.File(os.path.join(filepath_prefix, datasets[0]), "r") as f:
        data = f["train"]
        data = data - np.mean(data, axis=0)
        data = data / np.linalg.norm(data, axis=1, keepdims=True)

        mapper = umap.UMAP(
            n_neighbors=20,
        ).fit(data)
        # umap.plot.diagnostic(mapper, diagnostic_type='pca')
        # umap.plot.connectivity(mapper, edge_bundling='hammer')
        umap.plot.points(mapper)
        plt.show()
