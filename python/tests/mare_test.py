import h5py
import numpy as np
import pandas as pd
from pathlib import Path

from time import perf_counter
import sys
import os
sys.path.append(os.path.join(Path(__file__).resolve().parents[2]))

import panna
import argparse
from filelock import FileLock

if __name__ == "__main__":
    panna.set_seed(360)
    deltas = [0.01, 0.05, 0.1, 0.2]
    eps = [0.2, 0.5, 1.0, 5.0, 10.0, 20.0]
    paths = [
        "fashion-mnist-784-euclidean.hdf5",
         "glove-100-angular.hdf5",
         "nytimes-256-angular.hdf5",
         "gist-960-euclidean.hdf5",
         "simplewiki-openai-3072-normalized.hdf5",
         "sift-128-euclidean.hdf5",
         "deep-image-96-angular.hdf5"
    ]
    path_prefix = Path(__file__).resolve().parents[2]

    dataset_folder = os.path.join(path_prefix, "datasets")
    results_folder = os.path.join(path_prefix, "results")

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Dataset filename to process (optional)")
    args = parser.parse_args()

    if args.path:
        paths = [args.path]

    for path in paths:
        # Load dataset via new API and keep the same PCA behavior for PAMAP2
        stem = Path(path).stem
        _, data = panna.datasets.load(name=stem, pca_dimensions=4 if 'pamap2' in stem.lower() else None)
        data = np.array(data).astype(np.float32)[:1000]

        out_lines = []
        for delta in deltas:
            emst = panna.EMST(data, delta=delta, epsilon=0, family="lattice")
            start_time = perf_counter()
            edges = emst.find_mst()
            end_time = perf_counter()
            elapsed_time = end_time - start_time
            weight = sum(edges[0])
            out_lines.append(f"K+, {data.shape[0]}, {path}, {weight}, {elapsed_time}, {delta}\n")

            for epsilon in eps:
                emst = panna.EMST(data, delta=delta, epsilon=epsilon, family="lattice")
                start_time = perf_counter()
                edges = emst.find_mst()
                end_time = perf_counter()
                elapsed_time = end_time - start_time
                weight = sum(edges[0])
                out_lines.append(f"K± e{epsilon}, {data.shape[0]}, {path}, {weight}, {elapsed_time}, {delta}\n")

        # Append results under a file lock to avoid races between parallel jobs
        lock_path = os.path.join(results_folder, "weight_results.csv.lock")
        out_path = os.path.join(results_folder, "weight_results.csv")
        with FileLock(lock_path):
            with open(out_path, "a+") as f_out:
                for l in out_lines:
                    f_out.write(l)
                f_out.flush()
                
                
