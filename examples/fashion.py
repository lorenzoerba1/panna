import time
import numpy as np
import panna
import h5py
from icecream import ic

def compute_recall(ground, computed):
    k = computed.shape[0]
    thresh = ground[k-1]
    return np.mean(computed <= thresh)
    

with h5py.File("fashion-mnist-784-euclidean.hdf5") as hfp:
    data = hfp["/train"][:]
    queries = hfp["/test"][:]
    ground = hfp["/distances"][:]


panna.set_seed(1234)
index = panna.TrieIndex(data.shape[1], "euclidean", repetitions=32)
index.insert(data) 
index.rebuild()

print(index)
ic(index.num_repetitions, index.num_points)

k = 10
recalls = []
start = time.time()
for q_idx in range(1000):
    q = queries[q_idx]
    ans = index.search(q, k, 0.8)
    ans_data = data[ans]
    ans_dists = np.linalg.norm(ans_data - q, axis=1)
    recalls.append(compute_recall(ground[q_idx], ans_dists))
end = time.time()

elapsed = end - start
qps = len(recalls) / elapsed
ic(np.mean(recalls))
ic(elapsed)
ic(qps)
