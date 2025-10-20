#import fast_hdbscan
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
        #  "glove-100-angular.hdf5",
         "nytimes-256-angular.hdf5",
         "gist-960-euclidean.hdf5",
         "simplewiki-openai-3072-normalized.hdf5",
         "sift-128-euclidean.hdf5",
         "deep-image-96-angular.hdf5",
    ]
    path_prefix = Path(__file__).resolve().parents[1]

    dataset_folder = os.path.join(path_prefix, "datasets")
    results_folder = os.path.join(path_prefix, "results")
    
    with open(os.path.join(results_folder, "scalability_results.csv"), "a+") as f_out:
        for path in paths:
            with h5py.File(os.path.join(dataset_folder, path), "r") as f:
                data = np.array(f["train"]).astype(np.float32)
                
                emst = panna.EMST(data, delta= 0.01, epsilon=0.2)
                start_time = perf_counter()
                emst.find_exact_mst()
                end_time = perf_counter()
                elapsed_time = end_time - start_time
                f_out.write(f"K+, {data.shape[0]}, {path}, 0, {elapsed_time}\n")
                
                # emst = panna.EMST(data, delta= 0.01, epsilon=0.2)
                # start_time = perf_counter()
                # emst.find_epsilon_mst()
                # end_time = perf_counter()
                # elapsed_time = end_time - start_time
                # f_out.write(f"K± e0.2, {data.shape[0]}, {path}, 0, {elapsed_time}\n")
                f_out.flush()