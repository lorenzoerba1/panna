import h5py
import numpy as np
import pandas as pd
from pathlib import Path

from time import perf_counter
import sys
import os
sys.path.append(os.path.join(Path(__file__).resolve().parents[2]))

import _panna_impl as panna

if __name__ == "__main__":
    panna.set_seed(1989)
    deltas = [0.01, 0.05, 0.1, 0.2]
    eps = [0.2, 0.5, 1.0, 5.0, 10.0, 20.0]
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
    
    with open(os.path.join(results_folder, "weight_results.csv"), "a+") as f_out:
        for path in paths:
            with h5py.File(os.path.join(dataset_folder, path), "r") as f:
                data = np.array(f["train"]).astype(np.float32)[:5000]
                
                for delta in deltas:
                    
                    emst = panna.EMST(data, delta= delta, epsilon=0.2)
                    start_time = perf_counter()
                    weight = emst.find_exact_mst()
                    end_time = perf_counter()
                    elapsed_time = end_time - start_time
                    f_out.write(f"K+, {data.shape[0]}, {path}, {weight}, {elapsed_time}, {delta}\n")
                    
                    for epsilon in eps:
                        emst = panna.EMST(data, delta= delta, epsilon=epsilon)
                        start_time = perf_counter()
                        weight = emst.find_epsilon_mst()
                        end_time = perf_counter()
                        elapsed_time = end_time - start_time
                        # We have to write                     
                        # outfile <<"K± e" << ep << ", " << points.size() << ", " << name << ", " << weight << ", "<< duration << ", " << prob << std::endl;
                        f_out.write(f"K± e{epsilon}, {data.shape[0]}, {path}, {weight}, {elapsed_time}, {delta}\n")
                    f_out.flush()
                
                
