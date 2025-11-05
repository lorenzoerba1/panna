from sklearn.decomposition import PCA
import h5py
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path

from time import perf_counter
import sys
import os
sys.path.append(os.path.join(Path(__file__).resolve().parents[2]))

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
            "ethylene_CO.txt",
            "HT_Sensor_dataset.dat",
            "imagenet-align-640-normalized.hdf5",
            "landmark-nomic-768-normalized.hdf5",
            "9_census.npz",
            "PAMAP2_Dataset.zip",
    ]
    path_prefix = Path(__file__).resolve().parents[1]

    dataset_folder = os.path.join(path_prefix, "datasets")
    results_folder = os.path.join(path_prefix, "results")
    
    with open(os.path.join(results_folder, "scalability_results.csv"), "a+") as f_out:
        for path in paths:
            data = []
            if path.endswith(".hdf5"):
                with h5py.File(os.path.join(dataset_folder, path), "r") as f:
                    data = f["train"][:].astype(np.float32)
            elif path.endswith(".dat") or path.endswith(".txt"):
                data = pd.read_csv(os.path.join(dataset_folder, path), delim_whitespace=True)
                data = data.to_numpy().astype(np.float32)
                data = np.nan_to_num(data)
            elif path.endswith(".npz"):
                loaded = np.load(os.path.join(dataset_folder, path))
                data = loaded["X"].astype(np.float32)
            elif path.endswith(".zip"):
                with zipfile.ZipFile(os.path.join(dataset_folder, path), 'r') as zip_ref:
                    arr = []
                    for i in range(1,10):
                        zfn = f"PAMAP2_Dataset/Protocol/subject10{i}.dat"
                        zf = zip_ref.open(zfn)
                        for line in zf:
                            line = line.decode()
                            l = list(map(float, line.strip().split()))
                            # remove timestamp
                            arr.append(l[1:])
                    X = np.nan_to_num(np.array(arr)) # many NaNs in data, replace them with 0.
                    data = PCA(n_components=4).fit_transform(X) # PCA of first four components
              
            emst = panna.EMST(data, delta= 0.01, epsilon=0)
            start_time = perf_counter()
            emst.find_mst()
            end_time = perf_counter()
            elapsed_time = end_time - start_time
            f_out.write(f"K+, {data.shape[0]}, {path}, 0, {elapsed_time}\n")
            
            emst = panna.EMST(data, delta= 0.01, epsilon=0.2)
            start_time = perf_counter()
            emst.find_mst()
            end_time = perf_counter()
            elapsed_time = end_time - start_time
            f_out.write(f"K± e0.2, {data.shape[0]}, {path}, 0, {elapsed_time}\n")
            f_out.flush()