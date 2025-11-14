import panna
import panna.datasets
import numpy as np
import joblib
from icecream import ic
import csv
import logging
import matplotlib.pyplot as plt


MEM = joblib.Memory(".cache")


def compute_flexibility(tree, epsilon, diameter):
    total_cost = sum(tree)
    cost = 0
    for i, w in enumerate(tree):
        remaining = len(tree) - i
        cost += w
        lower_bound = remaining * w
        upper_bound = remaining * diameter
        # if cost + remaining * diameter <= (1 + epsilon) * total_cost:
        # if upper_bound - lower_bound <= epsilon * (cost + lower_bound):
        if upper_bound <= epsilon * cost:
            ic(cost, remaining, upper_bound, diameter)
            return remaining
    return 0


def compute_edge_mass(weights, counts, threshold):
    ic(threshold)
    idx = np.searchsorted(weights, threshold, side="right")
    ic(idx)
    return ic(counts[idx])


def compute_cumulative_distance_distribution(
    data, min_distance, max_distance, num_buckets=10000, sample_fraction=0.01
):
    n = data.shape[0]
    num_pairs = n * (n - 1) // 2
    samples = int(min(1e9, num_pairs * sample_fraction))
    counts, bounds = panna.distance_histogram(data, num_buckets, min_distance, max_distance, samples)
    counts = np.concatenate((counts, [0]))
    counts = np.cumsum(counts)
    return bounds, counts


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


def exact_emst(data):
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


@MEM.cache
def cached_emst(data):
    emst = panna.EMST(data, epsilon=0.0, delta=0.001).find_mst()
    our_weights = emst[0]
    recomputed_weights = [
        np.linalg.norm(data[e[0]] - data[e[1]])
        for e in emst[1]
    ]
    if data.shape[0] <= 10000:
        exact = panna.EMST(data).find_mst_exact()
        exact_weights = exact[0]
        tree, _, _ = exact_emst(data)
        check_weights = np.array([e[0] for e in tree])
        ic(recomputed_weights, our_weights, check_weights, exact_weights)
        ic(sum(recomputed_weights), sum(our_weights), sum(check_weights), sum(exact_weights))
        ic(emst[1][:10], tree[:10])
        assert sum(our_weights) == sum(check_weights)

    return emst


def compute_stats_csv():
    logging.basicConfig(level=logging.INFO)

    outfile = "examples/emst_stats.csv"
    with open(outfile, "w") as fp:
        out = csv.writer(fp)
        out.writerow(("dataset", "epsilon", "flexibility", "mass", "num_pairs", "mass_fraction"))

    datasets = panna.datasets.available_datasets()
    for dataset in datasets:
        if dataset != "fashion-mnist-784-euclidean":
            continue
        ic(dataset)
        pca_dimensions = 4 if dataset == "pamap2" else None
        _, data = panna.datasets.load(dataset, pca_dimensions=pca_dimensions)
        data = data[:100]
        n = data.shape[0]
        num_pairs = n * (n - 1) // 2
        weights, _edges = cached_emst(data)
        weights = np.sort(weights)
        diameter = panna.approximate_diameter(data)
        bounds, counts = compute_cumulative_distance_distribution(data, weights[0], diameter)

        plt.figure()
        plt.title(dataset)
        plt.plot(bounds, counts)
        
        # plt.plot(weights, [0.01]*len(weights), '|', color='k')
        with open(outfile, "a") as fp:
            out = csv.writer(fp)
            for epsilon in [0, 0.01, 0.1]:
                flexibility = compute_flexibility(weights, epsilon, diameter)
                max_rigid_weight = weights[-flexibility-1]
                mass = compute_edge_mass(bounds, counts, max_rigid_weight)
                plt.axhline(mass, linestyle="dotted")
                plt.axvline(max_rigid_weight, linestyle="dotted")
                plt.annotate(f"ε={epsilon}", (0, mass))
                out.writerow((dataset, epsilon, flexibility, mass, num_pairs, mass/num_pairs))
        plt.savefig(f"examples/cumdist-{dataset}.png", dpi=300)


compute_stats_csv()
