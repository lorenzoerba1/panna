import h5py
import os
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append("build")

if __name__ == "__main__":
    # For each dataset in the datasets folder, compute the mean weight of the edges, sampling 1000 points from the dataset
    datasets_folder = os.path.join(Path(__file__).resolve().parents[2], "datasets")
    datasets = [f for f in os.listdir(datasets_folder) if f.endswith(".hdf5")]
    for dataset in datasets:
        with h5py.File(os.path.join(datasets_folder, dataset), "r") as f:
            data = f["train"][:]
            if data.shape[0] > 1000:
                idx = np.random.choice(data.shape[0], 1000, replace=False)
                data = data[idx]
            # Compute all pairwise distances
            from sklearn.metrics import pairwise_distances

            distances = pairwise_distances(data, metric="euclidean")
            # Compute the mean of the distances
            mean_distance = np.mean(distances)
            print(f"Dataset: {dataset}, Mean distance: {mean_distance}")
