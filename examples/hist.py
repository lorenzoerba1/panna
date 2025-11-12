import panna
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import h5py
import joblib

MEM = joblib.Memory(".cache")

def compute_flexibility(tree, epsilon, diameter):
    total_cost = sum(tree)
    cost = 0
    for i, w in enumerate(tree):
        remaining = len(tree) - i
        cost += w
        if cost + remaining * diameter <= (1+epsilon) * total_cost:
            return remaining
    return 0



# with h5py.File("datasets/glove-100-angular.hdf5") as hfp:
with h5py.File("datasets/fashion-mnist-784-euclidean.hdf5") as hfp:
    data = hfp["train"][:]

n = data.shape[0]

weights, edges = panna.EMST(data, epsilon=0.0).find_mst()

bounds = weights

counts = panna.distance_histogram(data, bounds, 10_000_000)
counts = np.cumsum(counts)

diameter = panna.approximate_diameter(data)
print(diameter)


plt.plot(bounds, counts[1:])
plt.scatter(bounds, counts[1:])
plt.savefig("examples/hist.png")

plt.figure()
flexibility = compute_flexibility(weights, 0.1, diameter)
print(flexibility)
for i, w in enumerate(weights):
    plt.plot((i, i), (0, w), c="tab:blue" if i <= len(weights) - flexibility else "tab:red")

plt.savefig("examples/emst.png", dpi=300)


