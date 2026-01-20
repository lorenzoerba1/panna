import panna
import panna.datasets
import numpy as np
from icecream import ic
import time

rng = np.random.default_rng(1234)
_, data = panna.datasets.load("ht")
rng.shuffle(data)
data = data[:100000]
n = data.shape[0]
ic(data.shape)

start = time.time()
algo = panna.EMST(data, epsilon=0.0, delta=0.1, repetitions=32, family="lattice")
weight, emst = algo.find_mst()
end = time.time()
print("elapsed", end - start, "seconds", "weight", weight.sum())

