import panna
import panna.datasets
import numpy as np
from icecream import ic
import time

_, data = panna.datasets.load("fashion-mnist-784-euclidean")
data = data[:10000]
n = data.shape[0]

start = time.time()
algo = panna.EMST(data, epsilon=0.0, delta=0.1)
weight, emst = algo.find_mst()
end = time.time()
print("elapsed", end - start, "seconds")

