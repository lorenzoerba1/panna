import zipfile
from sklearn.decomposition import PCA
import h5py
import os
import numpy as np
import pandas as pd
from pathlib import Path
import sys


if __name__ == "__main__":
    paths = [
        #  "fashion-mnist-784-euclidean.hdf5",
        #    "glove-100-angular.hdf5",
        #   "nytimes-256-angular.hdf5",
        #  "gist-960-euclidean.hdf5",
        #   "simplewiki-openai-3072-normalized.hdf5",
        #   "sift-128-euclidean.hdf5",
        #   "deep-image-96-angular.hdf5",
        # "ethylene_CO.txt",
        # "HT_Sensor_dataset.dat",
        # "imagenet-align-640-normalized.hdf5",
        # "landmark-nomic-768-normalized.hdf5",
        "9_census.npz",
        "PAMAP2_Dataset.zip"
    ]
    path_prefix = Path(__file__).resolve().parents[1]

    dataset_folder = os.path.join(path_prefix, "datasets")
    results_folder = os.path.join(path_prefix, "results")
    
    with open(os.path.join(results_folder, "scalability_results_tutte.csv"), "a+") as f_out:
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
            
            if data.shape[0] > 1000:
                idx = np.random.choice(data.shape[0], 1000, replace=False)
                data = data[idx]
                    # Compute all pairwise distances
                from sklearn.metrics import pairwise_distances

                distances = pairwise_distances(data, metric="euclidean")
                # Order the distances
                distances_ordered = distances[np.triu_indices_from(distances, k=1)]
                
                # Use kruskal and UnionFind data structure to compute mst
                from scipy.sparse.csgraph import minimum_spanning_tree
                mst = minimum_spanning_tree(distances)
                
                # Mu last edge weight in the mst
                mu = mst.data[-1]
                # Two_mu edge at position 2* position of mu in the sorted distances
                # or the last edge if 2*position exceeds number of edges
                two_mu = distances_ordered[2 * np.searchsorted(distances_ordered, mu)] if 2 * np.searchsorted(distances_ordered, mu) < len(distances_ordered) else distances_ordered[-1]
                contrast = two_mu / mu
                
                # Compute the mst and print the contrast

                print(f"Dataset: {path}, Contrast: {contrast}")
