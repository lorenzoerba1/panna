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

import panna

if __name__ == "__main__":
    paths = [
                "fashion-mnist-784-euclidean",
                "glove-100-angular",
                "nytimes-256-angular",
                "gist-960-euclidean",
                "simplewiki-openai-3072-normalized",
                "sift-128-euclidean",
                "deep-image-96-angular",
                "chem",
                "ht",
                "imagenet-align-640-normalized",
                "landmark-nomic-768-normalized",
                "census",
                "pamap2",
        ]
    path_prefix = Path(__file__).resolve().parents[2]

    dataset_folder = os.path.join(path_prefix, "datasets")
    results_folder = os.path.join(path_prefix, "results")
    
    with open(os.path.join(results_folder, "scalability_results_mlpack.csv"), "a+") as f_out:
            for path in paths:
                _, data = panna.datasets.load(name=path, pca_dimensions=4 if path == "pamap2" else None)
                # Run on prefixes of 10,000 points
                
                start_time = perf_counter()
                mst = emst(data, verbose=False)
                end_time = perf_counter()
                elapsed_time = end_time - start_time
                f_out.write(f"mlpack-Boruvka, {data.shape[0]}, {path}, 0, {elapsed_time}\n")
                f_out.flush()