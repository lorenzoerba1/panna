import panna
import panna.datasets
import numpy as np
from icecream import ic
import time
from fast_hdbscan.hdbscan import compute_minimum_spanning_tree

rng = np.random.default_rng(1234)
_, data = panna.datasets.load("ht")
rng.shuffle(data)
data = data[:10000]
n = data.shape[0]
ic(data.shape)

start = time.time()
algo = panna.EMST(data, epsilon=0.0, delta=0.01, repetitions=32, family="e2lsh")
weight, emst = algo.find_mst()
xs = data[emst[:,0]]
ys = data[emst[:,1]]
our_weights = np.linalg.norm(xs - ys, axis=1)
end = time.time()
print("elapsed", end - start, "seconds", "weight", our_weights.sum())

start = time.time()
baseline = compute_minimum_spanning_tree(data, min_samples=1)
end = time.time()
edges = baseline[0].astype(np.int32)
xs = data[edges[:,0]]
ys = data[edges[:,1]]
baseline_weights = np.linalg.norm(xs - ys, axis=1)
assert ic(baseline_weights.shape[0]) == n - 1
print("baseline elapsed", end - start, "baseline weight", baseline_weights.sum())
