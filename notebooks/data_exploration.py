import marimo

__generated_with = "0.23.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import panna
    import panna.datasets
    import numpy as np
    import fast_hdbscan
    import time

    return fast_hdbscan, mo, np, panna, time


@app.cell
def _(np, panna):
    def load(dataset="gist-960-euclidean", sample_frac=0.01):
        _, data = panna.datasets.load(dataset)
        if sample_frac is None:
            return data
        sample_seed = 1234
        sample_size = int(sample_frac * data.shape[0])
        print(f"sampling {sample_size} elements")
        rng = np.random.default_rng(sample_seed)
        indices = rng.choice(data.shape[0], sample_size)
        data = data[indices]
        print("loaded data")
        return data

    data = load("fashion-mnist-784-euclidean", sample_frac=None)
    return (data,)


@app.cell
def _():
    import matplotlib.pyplot as plt

    return (plt,)


@app.cell
def _(data, panna):
    num_buckets = 1000
    min_distance=0.0
    max_distance=12000.0
    samples=1000000
    counts, bounds = panna.distance_histogram(
        data, num_buckets, min_distance, max_distance, samples
    )
    return bounds, counts


@app.cell
def _(bounds, counts, distances, plt):
    maxsampled = [(c, b) for c, b in zip(counts, bounds) if c > 0][-1]
    plt.plot(bounds, counts)
    plt.axvline(maxsampled[1], color="red")
    plt.axvline(distances.max(), color="green")
    plt.annotate("max tree edge", xy=(distances.max(), 50000))
    plt.annotate("diameter", xy=(maxsampled[1], 50000))
    return


@app.cell
def _(data, fast_hdbscan, mo, time):
    @mo.cache
    def compute_emst(data):
        start = time.time()
        res = fast_hdbscan.hdbscan.compute_minimum_spanning_tree(data, min_samples=0)
        end = time.time()
        return res, end - start

    res, fast_hdbscan_elapsed_s = compute_emst(data)
    return (res,)


@app.cell
def _(res):
    spanning_tree, neighbors, core_distances = res
    return (spanning_tree,)


@app.cell
def _(spanning_tree):
    weights = spanning_tree[:,2]
    weights.max()
    return


@app.cell
def _(data, np, spanning_tree):
    xs = data[spanning_tree[:,0].astype(np.int32)]
    ys = data[spanning_tree[:,1].astype(np.int32)]
    distances = np.linalg.norm(xs - ys, axis=1)
    distances.sort()
    return (distances,)


@app.cell
def _(distances):
    distances.max() / distances.mean()
    return


@app.cell
def _(distances, plt):
    plt.plot(distances)
    return


@app.cell
def _(data, panna):
    algo = panna.EMST(data, epsilon=0.0, delta=0.1, repetitions=512, family="lattice")
    myemst = algo.find_mst()
    return


if __name__ == "__main__":
    app.run()
