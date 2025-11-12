import panna
import panna.datasets
import numpy as np
import matplotlib.pyplot as plt
import joblib

MEM = joblib.Memory(".cache")


def compute_flexibility(tree, epsilon, diameter):
    total_cost = sum(tree)
    cost = 0
    for i, w in enumerate(tree):
        remaining = len(tree) - i
        cost += w
        if cost + remaining * diameter <= (1 + epsilon) * total_cost:
            return remaining
    return 0


@MEM.cache
def cached_emst(data):
    return panna.EMST(data, epsilon=0.0).find_mst()


_, data = panna.datasets.load("fashion-mnist-784-euclidean")

weights, edges = cached_emst(data)

counts = panna.distance_histogram(data, weights, 10_000_000)
counts = np.cumsum(counts)

diameter = panna.approximate_diameter(data)
print(diameter)


plt.plot(weights, counts[1:])
plt.scatter(weights, counts[1:])
plt.savefig("examples/hist.png")

plt.figure()
flexibility = compute_flexibility(weights, 0.1, diameter)
print(flexibility)
for i, w in enumerate(weights):
    plt.plot(
        (i, i), (0, w), c="tab:blue" if i <= len(weights) - flexibility else "tab:red"
    )

plt.savefig("examples/emst.png", dpi=300)
