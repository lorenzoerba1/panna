import panna
import panna.datasets
import numpy as np
from icecream import ic
import time

_, data = panna.datasets.load("ht")
# data = data[:100000]
n = data.shape[0]
ic(data.shape)

start = time.time()
algo = panna.EMST(data, epsilon=0.0, delta=0.1, repetitions=64)
weight, emst = algo.find_mst()
end = time.time()
print("elapsed", end - start, "seconds")

