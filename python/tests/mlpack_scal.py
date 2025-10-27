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
import time
import sys
from mlpack import emst

sys.path.append("build")
import _panna_impl as panna


if __name__ == "__main__":
    data_size = [10**i for i in range(4, 7)]
    dim_size = [10]

    for n in data_size:
        for d in dim_size:
            print(f"Data size: {n}, Dimension: {d}")
            data = np.random.rand(n, d).astype(np.float32)

            start_time = time.perf_counter()
            emst(data, verbose=False)
            end_time = time.perf_counter()

            elapsed_time = end_time - start_time
            print(f"Elapsed time for EMST: {elapsed_time:.4f} seconds\n")
