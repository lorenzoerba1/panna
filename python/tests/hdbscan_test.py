import fast_hdbscan
import h5py
import numpy as np
import pandas as pd
import sklearn.datasets as data
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
)
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import umap
from time import perf_counter
import sys
import os
from pathlib import Path
#from mlpack import emst

sys.path.append(os.path.join(Path(__file__).resolve().parents[2]))
import _panna_impl as panna
#from keras.datasets import fashion_mnist


def unpickle(file):
    import pickle

    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


if __name__ == "__main__":
    matplotlib.use("WebAgg")
    # Open and read the dataset
    with h5py.File(
        "/home/monaco/span/panna/datasets/fashion-mnist-784-euclidean.hdf5", "r"
    ) as f:
       # test_data = pd.read_csv("datasets/ethylene_CO.txt", delim_whitespace=True)
        # print(test_data.head(), test_data.shape)
        # test_data = test_data.to_numpy()
        #test_data = f["train"][:5000]
        # Choose a subset of the data for testing
        # (_, true_labels), _ = fashion_mnist.load_data()
        # true_labels = true_labels[:5000]
        #     print("Data shape:", data.shape)

        test_data, true_labels = data.load_wine(return_X_y=True)
        # Cluster with both our version and the classic one
        # 1. Fast HDBSCAN
        start_time = perf_counter()
        proc = fast_hdbscan.HDBSCAN(
            min_samples=15,
            cluster_selection_method="eom",
            min_cluster_size=50,
            max_cluster_size=500,
            plus=False,
        )
        y = proc.fit_predict(test_data)
        end_time = perf_counter()
        elapsed_time = end_time - start_time

        print("Elapsed time for Fast HDBSCAN:", elapsed_time, "seconds")

        # 2. K+ HDBSCAN
        start_time = perf_counter()
        proc2 = fast_hdbscan.HDBSCAN(
            min_samples=15,
            cluster_selection_method="eom",
            min_cluster_size=50,
            max_cluster_size=500,
            plus=True,
        )
        y2 = proc2.fit_predict(test_data)
        end_time = perf_counter()
        elapsed_time = end_time - start_time
        print("Elapsed time for K+ HDBSCAN:", elapsed_time, "seconds")

        # If the dimensionality is greater than 2, reduce it to 2D using UMAP for visualization, otherwise use the data as is
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        plt.suptitle("HDBSCAN vs K+ HDBSCAN")
        plt.subplot(1, 2, 1)
        if test_data.shape[1] > 5:
            # Plot the clusters
            mapper = umap.UMAP().fit(test_data[:2000])
            reduced_data = mapper.transform(test_data[:2000])
            # Map labels to markers
            sns.scatterplot(
                x=reduced_data[:, 0],
                y=reduced_data[:, 1],
                hue=y[:2000],
                style=y[:2000],
                palette="tab10",
                s=20,
                alpha=0.7,
            )
        else:
            sns.scatterplot(
                x=test_data[:, 0],
                y=test_data[:, 1],
                hue=y,
                style=y,
                palette="tab10",
                s=20,
                alpha=0.7,
            )
        plt.title("HDBSCAN")
        plt.subplot(1, 2, 2)
        if test_data.shape[1] > 5:
            # Map labels to markers
            sns.scatterplot(
                x=reduced_data[:, 0],
                y=reduced_data[:, 1],
                hue=y2[:2000],
                style=y2[:2000],
                palette="tab10",
                s=20,
                alpha=0.7,
            )
        else:
            sns.scatterplot(
                x=test_data[:, 0],
                y=test_data[:, 1],
                hue=y2,
                style=y2,
                palette="tab10",
                s=20,
                alpha=0.7,
            )
        plt.title("K+ HDBSCAN")
        plt.show()

        # Compute silhouette score for both approaches, we drop -1 labels (noise)
        if len(set(y)) > 1 and len(set(y2)) > 1:
            sil_score = silhouette_score(test_data[y != -1], y[y != -1])
            sil_score2 = silhouette_score(test_data[y2 != -1], y2[y2 != -1])
            print("Silhouette Score HDBSCAN:", sil_score)
            print("Silhouette Score K+ HDBSCAN:", sil_score2)
        if len(true_labels) > 0:
        # Compute ARI and AMI using ground truth
            ari = adjusted_rand_score(true_labels, y)
            ami = adjusted_mutual_info_score(true_labels, y)
            print("ARI S :", ari)
            print("AMI S :", ami)
            ari2 = adjusted_rand_score(true_labels, y2)
            ami2 = adjusted_mutual_info_score(true_labels, y2)
            print("ARI K+ :", ari2)
            print("AMI K+ :", ami2)

        # Compute the MST with mlpack for comparison
        start_time = perf_counter()
        mst_input = pd.DataFrame(test_data)
        mst = emst(input_=mst_input, verbose=True)
        end_time = perf_counter()
        elapsed_time = end_time - start_time
        print("Elapsed time for MST (mlpack):", elapsed_time, "seconds")
