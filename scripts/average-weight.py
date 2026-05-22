#!/usr/bin/env python
"""
For each dataset in ``panna.datasets.available_datasets()``, estimate the
average pairwise distance by sampling a few million random pairs of points.

For datasets with ``angular`` distance the data is L2-normalized first, so the
reported value is the average Euclidean distance between unit vectors (which is
monotone in cosine distance and matches what the EMST scripts in this repo
use as edge weight).
"""

import argparse
import logging
import sys
import time

import numpy as np

from panna import datasets


def average_pairwise_distance(
    X: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    """Estimate E[||X_i - X_j||_2] by sampling ``n_samples`` random pairs.

    Returns (mean, standard error of the mean, pairs actually used).
    """
    n = X.shape[0]
    if n < 2:
        raise ValueError(f"need at least 2 points, got {n}")

    # Cap per-batch memory at ~200M float32 values for the (i - j) intermediate.
    dim = X.shape[1]
    batch_size = int(max(1000, min(200_000, 200_000_000 // max(1, dim))))

    total = 0.0
    total_sq = 0.0
    used = 0
    while used < n_samples:
        b = min(batch_size, n_samples - used)
        i = rng.integers(0, n, size=b)
        j = rng.integers(0, n, size=b)
        mask = i != j
        if not mask.all():
            i, j = i[mask], j[mask]
        if i.size == 0:
            continue
        diff = X[i].astype(np.float64, copy=False) - X[j].astype(np.float64, copy=False)
        d = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        total += float(d.sum())
        total_sq += float(np.dot(d, d))
        used += int(d.size)

    mean = total / used
    var = max(total_sq / used - mean * mean, 0.0)
    stderr = float(np.sqrt(var / used))
    return mean, stderr, used


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        type=int,
        default=2_000_000,
        help="Number of pairs to sample per dataset (default: 2_000_000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="PRNG seed (default: 1234).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional CSV file to write the results to (in addition to stdout).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Restrict to the given dataset names (default: all available).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    rng = np.random.default_rng(args.seed)

    names = args.datasets if args.datasets else datasets.available_datasets()

    header = "dataset,distance,n,dim,n_samples,mean_distance,stderr,elapsed_s"
    print(header, flush=True)
    rows = [header]

    for name in names:
        try:
            pca_dimensions = 4 if "pamap2" in name.lower() else None
            distance, X = datasets.load(name, pca_dimensions=pca_dimensions)

            logging.info(
                "sampling %d pairs from %s (n=%d, dim=%d, distance=%s)",
                args.samples,
                name,
                X.shape[0],
                X.shape[1],
                distance,
            )
            t0 = time.perf_counter()
            mean, stderr, used = average_pairwise_distance(X, args.samples, rng)
            elapsed = time.perf_counter() - t0

            row = (
                f"{name},{distance},{X.shape[0]},{X.shape[1]},{used},"
                f"{mean:.6f},{stderr:.6e},{elapsed:.2f}"
            )
            print(row, flush=True)
            rows.append(row)
        except Exception as exc:
            logging.exception("failed on %s", name)
            row = f"{name},ERROR,,,,,{type(exc).__name__}: {exc},"
            print(row, flush=True)
            rows.append(row)

    if args.output:
        with open(args.output, "w") as fp:
            fp.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    sys.exit(main())
