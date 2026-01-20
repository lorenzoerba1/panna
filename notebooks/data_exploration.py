import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import panna
    import panna.datasets
    import numpy as np
    return mo, np, panna


@app.cell
def _(panna):
    _, data = panna.datasets.load("ht")

    return (data,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    return (plt,)


@app.cell
def _(data, panna):
    num_buckets = 1000
    min_distance=0.0
    max_distance=20000.0
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
    return


@app.cell
def _():
    import fast_hdbscan
    return (fast_hdbscan,)


@app.cell
def _(data, fast_hdbscan, mo):
    @mo.cache
    def compute_emst(data):
        return fast_hdbscan.hdbscan.compute_minimum_spanning_tree(data, min_samples=0)

    res = compute_emst(data)
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


if __name__ == "__main__":
    app.run()
