import panna
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import h5py


def compute_mu(weights, counts, epsilon, diameter):
    mu = 0
    cost = 0
    for i, (w, c) in enumerate(zip(weights, counts[1:])):
        remaining = len(weights) - i
        upper = remaining * diameter
        lower = remaining * w
        cost += w
        mu += c
        if (upper - lower) <= epsilon * cost:
            return i, mu


def compute_uncertainty(tree, epsilon, diameter):
    cost = 0
    for i, w in enumerate(tree):
        remaining = len(tree) - i
        cost += w
        lower = remaining * w
        upper = remaining * diameter
        if upper - lower <= epsilon * (cost):
            return i
    return len(tree)


with h5py.File("datasets/glove-100-angular.hdf5") as hfp:
# with h5py.File("datasets/fashion-mnist-784-euclidean.hdf5") as hfp:
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

for eps in [0.1]:
    mui, mu = compute_mu(weights, counts, eps, diameter)
    plt.axvline(weights[mui], c="tab:orange")

plt.savefig("examples/hist.png")

plt.figure()
unc = compute_uncertainty(weights, 0.1, diameter)
print(unc)
for i, w in enumerate(weights):
    plt.plot((i, i), (0, w), c="tab:blue" if i <= unc else "lightgray")
for i in range(unc, len(weights)):
    plt.plot((i, i), (0, weights[unc]), c="cyan")
    
plt.gca().add_patch(
    Rectangle(
        (unc, weights[unc]), len(weights) - unc, diameter - weights[unc], color="tab:orange"
    )
)

plt.savefig("examples/emst.png", dpi=300)


