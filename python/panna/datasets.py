"""
Datasets that can be used to try out different algorithms.
"""

import numpy as np
import os
from pathlib import Path
import urllib.request
from urllib.parse import urlparse
import logging
import h5py
import zipfile
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


DATASETS_DIR = Path(os.environ.get("PANNA_DATA_DIR", "datasets"))


def _download(url, destination: Path):
    if not destination.is_file():
        logging.info(f"downloading {url} to {destination}")
        urllib.request.urlretrieve(url, destination)


def _load_hdf5(path: Path):
    with h5py.File(path) as hfp:
        return hfp["train"][:], hfp["test"][:], hfp["distances"][:]


def _load_pamap(path: Path):
    def load_zipfile():
        with zipfile.ZipFile(path, "r") as zip_ref:
            arr = []
            for i in range(1, 10):
                zfn = f"PAMAP2_Dataset/Protocol/subject10{i}.dat"
                zf = zip_ref.open(zfn)
                for line in zf:
                    line = line.decode()
                    l = list(map(float, line.strip().split()))
                    # remove timestamp
                    arr.append(l[1:])
            X = np.nan_to_num(np.array(arr))  # many NaNs in data, replace them with 0.
            return X.astype(np.float32)

    h5path = path.parent / "pamap.hdf5"
    if not h5path.is_file():
        data = load_zipfile()
        with h5py.File(h5path, "w") as hfp:
            hfp["X"] = data

    with h5py.File(h5path) as hfp:
        data = hfp["X"][:]

    return data, None, None


def _load_census(path: Path):
    raw = np.load(path)
    data = raw["X"].astype(np.float32)
    return data, None, None


_DATASETS_INFO = {
    "landmark-nomic-768-normalized": (
        "https://huggingface.co/datasets/vector-index-bench/vibe/blob/main/landmark-nomic-768-normalized.hdf5",
        _load_hdf5,
        "euclidean",
    ),
    "imagenet-clip-512-normalized": (
        "https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/imagenet-clip-512-normalized.hdf5",
        _load_hdf5,
        "euclidean",
    ),
    "simplewiki-openai-3072-normalized": (
        "https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/simplewiki-openai-3072-normalized.hdf5",
        _load_hdf5,
        "euclidean",
    ),
    "deep-image-96-angular": (
        "http://ann-benchmarks.com/deep-image-96-angular.hdf5",
        _load_hdf5,
        "angular",
    ),
    "fashion-mnist-784-euclidean": (
        "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5",
        _load_hdf5,
        "euclidean",
    ),
    "glove-100-angular": (
        "http://ann-benchmarks.com/glove-100-angular.hdf5",
        _load_hdf5,
        "angular",
    ),
    "gist-960-euclidean": (
        "http://ann-benchmarks.com/gist-960-euclidean.hdf5",
        _load_hdf5,
        "euclidean",
    ),
    "nytimes-256-angular": (
        "http://ann-benchmarks.com/nytimes-256-angular.hdf5",
        _load_hdf5,
        "angular",
    ),
    "sift-128-euclidean": (
        "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
        _load_hdf5,
        "euclidean",
    ),
    "pamap2": (
        "http://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip",
        _load_pamap,
        "euclidean",
    ),
    "census": (
        "https://github.com/Minqi824/ADBench/raw/main/adbench/datasets/Classical/9_census.npz",
        _load_census,
        "euclidean",
    ),
    # "": "https://archive.ics.uci.edu/static/public/362/gas+sensors+for+home+activity+monitoring.zip",
    # "": "https://archive.ics.uci.edu/static/public/322/gas+sensor+array+under+dynamic+gas+mixtures.zip",
}


def available_datasets():
    return list(_DATASETS_INFO.keys())


def load(name: str, pca_dimensions=None, center_mean=False, load_queries=False):
    if name not in available_datasets():
        raise KeyError(
            f"Dataset `{name}` not available. Pick one of {available_datasets()}"
        )
    if not DATASETS_DIR.is_dir():
        DATASETS_DIR.mkdir()

    url, loader, distance = _DATASETS_INFO[name]
    local_name = DATASETS_DIR / Path(urlparse(url).path).name
    _download(url, local_name)
    train, test, distances = loader(local_name)

    if center_mean:
        scaler = StandardScaler(with_std=False)
        train = scaler.fit_transform(train)
        if test is not None:
            test = scaler.transform(test)

    if pca_dimensions is not None:
        pca = PCA(n_components=pca_dimensions)
        train = pca.fit_transform(train)
        if test is not None:
            test = pca.fit_transform(test)

    if load_queries:
        return distance, train, test, distances
    else:
        return distance, train


