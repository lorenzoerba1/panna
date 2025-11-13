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


@MEM.cache
def cached_emst(data):
    return panna.EMST(data, epsilon=0.0).find_mst()


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
        n = data.shape[0]
        num_pairs = n * (n - 1) // 2
        weights, edges = cached_emst(data)
        weights = np.sort(weights)
        diameter = panna.approximate_diameter(data)
        bounds, counts = compute_cumulative_distance_distribution(data, weights[0], diameter)
        # exact_counts, exact_bounds = exact_hist(data)

        plt.figure()
        plt.plot(bounds, counts)
        # plt.plot(exact_bounds, exact_counts, c="red")
        with open(outfile, "a") as fp:
            out = csv.writer(fp)
            for epsilon in [0, 0.01, 0.1]:
                flexibility = compute_flexibility(weights, epsilon, diameter)
                mass = compute_edge_mass(bounds, counts, weights[-flexibility - 1])
                plt.axhline(mass, linestyle="dotted")
                plt.annotate(f"ε={epsilon}", (0, mass))
                out.writerow((dataset, epsilon, flexibility, mass, num_pairs, mass/num_pairs))
        plt.savefig(f"examples/cumdist-{dataset}.png", dpi=300)


compute_stats_csv()
