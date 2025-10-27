import pyhdbscan
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import sklearn.datasets as data_loader
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
)
from scipy.cluster.hierarchy import fcluster

from time import perf_counter
import sys
import os
sys.path.append(os.path.join(Path(__file__).resolve().parents[2]))

if __name__ == "__main__":
    # Test on basic datasets
    wine_data, wine_labels = data_loader.load_wine(return_X_y=True)
    iris_data, iris_labels = data_loader.load_iris(return_X_y=True)
    diabetes_data, diabetes_labels = data_loader.load_diabetes(return_X_y=True)
    datasets = [
        (wine_data, wine_labels, "Wine"),
        (iris_data, iris_labels, "Iris"),
        (diabetes_data, diabetes_labels, "Diabetes"),
    ]
    
    # with open("results/hdbscan_sigmod_results.csv", "a+") as f_out:
    #     for data, true_labels, name in datasets:
    data = pd.read_csv("datasets/ethylene_CO.txt", delim_whitespace=True)
    start = perf_counter()
    clusterer = pyhdbscan.HDBSCAN(data, minPts=10)
    # labels = fcluster(clusterer, 1.0)
    elapsed_time = perf_counter() - start
    
    # ami = adjusted_mutual_info_score(true_labels, labels)
    # ari = adjusted_rand_score(true_labels, labels)
    
    #print(f"Dataset: {name}")
    print(f"Time taken: {elapsed_time:.4f} seconds")
            # print(f"Adjusted Mutual Info Score: {ami:.4f}")
            # print(f"Adjusted Rand Index: {ari:.4f}")
            # f_out.write(f"{name}, {data.shape[0]}, {elapsed_time}, {ami}, {ari}\n")
            # f_out.flush()
    
