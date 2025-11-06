import panna
import numpy as np
import matplotlib.pyplot as plt
import h5py

with h5py.File("datasets/fashion-mnist-784-euclidean.hdf5") as hfp:
    data = hfp["train"][:]

n = data.shape[0]
# gen = np.random.default_rng(1)
# data = gen.uniform(0, 1, size=(n, 2))

weights, edges = panna.EMST(data, epsilon=0.0).find_mst()
print(weights)
print(edges)

bounds = weights

counts = panna.distance_histogram(data, bounds, 10_000_000)
counts = np.cumsum(counts)
print(bounds)
print(counts)
print(counts.sum(), n*(n-1) // 2)
print(bounds.shape, counts.shape)

plt.plot(bounds, counts[1:])
plt.scatter(bounds, counts[1:])
plt.savefig("examples/hist.png")


