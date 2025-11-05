#import mlpack
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

from time import perf_counter
import sys
import os
sys.path.append(os.path.join(Path(__file__).resolve().parents[2]))
from mlpack import emst

#import panna

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
    path_prefix = Path(__file__).resolve().parents[2]

    dataset_folder = os.path.join(path_prefix, "datasets")
    results_folder = os.path.join(path_prefix, "results")
    
    with open(os.path.join(results_folder, "scalability_results_mlpack.csv"), "a+") as f_out:
        for path in paths:
            with h5py.File(os.path.join(dataset_folder, path), "r") as f:
                # Run on prefixes of 10,000 points
                data = np.array(f["train"]).astype(np.float32)[:10000]
                
                start_time = perf_counter()
                mst = emst(data, verbose=False)
                end_time = perf_counter()
                elapsed_time = end_time - start_time
                f_out.write(f"mlpack-Boruvka, {data.shape[0]}, {path}, 0, {elapsed_time}\n")
                f_out.flush()