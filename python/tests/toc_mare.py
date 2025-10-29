import h5py
import numpy as np
import pandas as pd
from pathlib import Path

from time import perf_counter
import sys
import os
sys.path.append(os.path.join(Path(__file__).resolve().parents[2]))

import panna

if __name__ == "__main__":
    panna.set_seed(1989)
    deltas = [0.1, 0.2]
    gammas = [0.2, 0.5, 1.0]
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

    with open("toc_weight_results.csv", "a+") as f_out:
        for path in paths:
            with h5py.File(os.path.join(dataset_folder, path), "r") as f:
                data = np.array(f["train"]).astype(np.float32)[:5000]
                
                for delta in deltas:
                    for gamma in gammas:
                        start_time = perf_counter()
                        weight = panna.emst_theory_of_computing(data, delta=delta, gamma=gamma)
                        weight = np.sum(weight[0])
                        end_time = perf_counter()
                        elapsed_time = end_time - start_time
                        # We have to write                     
                        # outfile <<"K± e" << ep << ", " << points.size() << ", " << name << ", " << weight << ", "<< duration << ", " << prob << std::endl;
                        f_out.write(f"ToC-K γ {gamma} c {delta}, {data.shape[0]}, {path}, {weight}, {elapsed_time}\n")
                    f_out.flush()