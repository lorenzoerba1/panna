import pyhdbscan
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

from time import perf_counter
import sys
import os
sys.path.append(os.path.join(Path(__file__).resolve().parents[2]))
#from mlpack import emst

import panna

if __name__ == "__main__":
    paths = [
        "fashion-mnist-784-euclidean.hdf5",
        "glove-100-angular.hdf5",
         "nytimes-256-angular.hdf5",
         "gist-960-euclidean.hdf5",
         "simplewiki-openai-3072-normalized.hdf5",
         "sift-128-euclidean.hdf5",
         "deep-image-96-angular.hdf5",
    ]
    path_prefix = Path(__file__).resolve().parents[1]

    dataset_folder = os.path.join(path_prefix, "datasets")
    results_folder = os.path.join(path_prefix, "results")
    
    dimensions = [5, 10, 20, 50] #, 128, 1024]
    size = [10**7]#10**4, 10**5, 10**6]
    
    with open(os.path.join(results_folder, "scalability_results_pyhdbscan.csv"), "a+") as f_out:
        for dim in dimensions:
            for sz in size:
                data = np.random.rand(sz, dim).astype(np.float32)
                
                start = perf_counter()
                pyhdbscan.HDBSCAN(data, 1)
                elapsed_time = perf_counter() - start
                
                print(f"Dimension: {dim}, Size: {sz}, Time: {elapsed_time:.2f} seconds")
                f_out.write(f"Wang-GFK,{size},{dim},{elapsed_time}\n")

    # with open(os.path.join(results_folder, "scalability_results.csv"), "a+") as f_out:
    #     for path in paths:
    #         with h5py.File(os.path.join(dataset_folder, path), "r") as f:
    #             data = np.array(f["train"]).astype(np.float32)
                
    #             start = perf_counter()
