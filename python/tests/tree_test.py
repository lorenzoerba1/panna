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
from mlpack import emst

sys.path.append("build")
import _panna_impl as panna
from keras.datasets import fashion_mnist


def unpickle(file):
    import pickle

    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


if __name__ == "__main__":
    with h5py.File(
        "/home/monaco/span/panna/datasets/sift-128-euclidean.hdf5", "r"
    ) as f:
        test_data = f["train"]  # [:5000]

        # Tufte parallel boruvka
        start_time = perf_counter()
        tree = fast_hdbscan.compute_minimum_spanning_tree(test_data, 1, None, False)
        end_time = perf_counter()
        elapsed_time = end_time - start_time
        print("Elapsed time for Fast HDBSCAN EMST:", elapsed_time, "seconds")
