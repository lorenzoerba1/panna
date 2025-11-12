import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.datasets import make_blobs
from icecream import ic


class UnionFind:
    def __init__(self, size):
        # Initially, each element is its own parent (self root)
        self.parent = list(range(size))
        # Rank used for union by rank optimization
        self.rank = [0] * size

    def find(self, x):
        # Path compression heuristic
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        # Union by rank heuristic
        rootX = self.find(x)
        rootY = self.find(y)

        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1

    def connected(self, x, y):
        # Check if two elements are in the same set
        return self.find(x) == self.find(y)


def compute_emst(data):
    # from fast_hdbscan.hdbscan import compute_minimum_spanning_tree
    # return compute_minimum_spanning_tree(data, min_samples=1)
    n = data.shape[0]
    edges = []
    for i in range(n):
        for j in range(i):
            edges.append((np.linalg.norm(data[i] - data[j]), i, j))

    tree = []
    edges = sorted(edges)
    uf = UnionFind(n)
    for edge in edges:
        if not uf.connected(edge[1], edge[2]):
            uf.union(edge[1], edge[2])
            tree.append(edge)

    diameter = edges[-1][0]

    return tree, edges, diameter


def compute_mu(tree, epsilon, diameter):
    cost = 0
    for i, (w, _, _) in enumerate(tree):
        remaining = len(tree) - i
        cost += w
        lower = remaining * w
        upper = remaining * diameter
        if upper - lower <= epsilon * (cost):
            return i
    return len(tree)


def compute_flexibility(tree, epsilon, diameter):
    total_cost = sum([e[0] for e in tree])
    cost = 0
    for i, (w, _, _) in enumerate(tree):
        remaining = len(tree) - i
        ic(remaining)
        cost += w
        if ic(cost) + ic(remaining * diameter) <= ic((1+epsilon) * total_cost):
            return remaining
    return 0



n = 100
epsilon = 1
gen = np.random.default_rng(1234)
# data = gen.uniform(0, 10, size=(n, 2))
data, _ = make_blobs(n, cluster_std=0.5, centers=20, random_state=1234)
# radii = gen.uniform(0, 1, size=n)
# # radii = np.ones(n)
# directions = gen.normal(size=(n, 2))
# directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]
# data = directions * radii[:, np.newaxis]

height = 3
fig, axs = plt.subplots(1, 2, figsize=(height*4, height), width_ratios=(1, 4))

tree, edges, diameter = compute_emst(data)
print(diameter)
flexibility = compute_flexibility(tree, epsilon, diameter)
print(flexibility)

for i, (w, parent, child) in enumerate(tree):
    is_arbitrary = i >= len(tree) - flexibility
    axs[0].plot(
        [data[parent, 0], data[child, 0]],
        [data[parent, 1], data[child, 1]],
        c="tab:red" if is_arbitrary else "tab:blue",
        # alpha=0.5,
    )
    axs[1].plot([i, i], [0, w], c="tab:red" if is_arbitrary else "tab:blue")

# axs[1].axvline(flexibility)
# padding = 0
# axs[1].add_patch(
#     Rectangle(
#         (len(tree) - flexibility - padding, 0),
#         (flexibility-1) + padding,
#         diameter,
#         color="tab:orange",
#     )
# )
# axs[1].set_ylim(0, diameter * 1.01)
axs[1].axis("off")

axs[0].scatter(data[:, 0], data[:, 1], s=5, c="black", zorder=100)
axs[0].axis("off")
plt.tight_layout()
plt.savefig("examples/flexibility-example.png")
