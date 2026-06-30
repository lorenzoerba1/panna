import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm



def bucket_to_string(codes):

    parts = []

    for c in codes:

        c = np.int8(c)

        parts.append(
            format(np.uint32(np.int32(c)), "x")
        )

    return "#" + "_".join(parts)




def draw_hash_functions(ax, hashes, xmin, xmax, ymin, ymax):

    x = np.linspace(xmin, xmax, 400)

    for _, row in hashes.iterrows():

        a1 = row["ax"]
        a2 = row["ay"]
        b = row["offset"]
        w = row["width"]

        # calcola quali valori di k possono intersecare la finestra
        corners = [
            a1*xmin + a2*ymin + b,
            a1*xmin + a2*ymax + b,
            a1*xmax + a2*ymin + b,
            a1*xmax + a2*ymax + b
        ]

        kmin = int(np.floor(min(corners)/w)) - 1
        kmax = int(np.ceil(max(corners)/w)) + 1

        for k in range(kmin, kmax+1):

            c = k*w - b

            # retta verticale
            if abs(a2) < 1e-8:

                xv = c / a1

                if xmin <= xv <= xmax:
                    ax.plot(
                        [xv, xv],
                        [ymin, ymax],
                        color="blue",
                        linewidth=0.7,
                        alpha=0.5
                    )

            # retta normale
            else:

                y = (c - a1*x)/a2

                mask = (y >= ymin) & (y <= ymax)

                if np.any(mask):
                    ax.plot(
                        x[mask],
                        y[mask],
                        color="blue",
                        linewidth=0.7,
                        alpha=0.5
                    )



# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

dataset = pd.read_csv("Entropy_NN/dataset.csv")
query = pd.read_csv("Entropy_NN/query.csv")
hashes = pd.read_csv("Entropy_NN/hash_functions.csv")
bucket_visits = pd.read_csv("Entropy_NN/bucket_visits.csv")

# bucket -> visits dictionary
visits = dict(zip(bucket_visits["bucket"], bucket_visits["visits"]))

# ------------------------------------------------------------
# Plot limits
# ------------------------------------------------------------

margin = 1.0

xmin = min(dataset["x"].min(), query["x"][0]) - margin
xmax = max(dataset["x"].max(), query["x"][0]) + margin

ymin = min(dataset["y"].min(), query["y"][0]) - margin
ymax = max(dataset["y"].max(), query["y"][0]) + margin

# ------------------------------------------------------------
# Dense grid
# ------------------------------------------------------------

resolution = 500

xs = np.linspace(xmin, xmax, resolution)
ys = np.linspace(ymin, ymax, resolution)

heat = np.zeros((resolution, resolution))

# ------------------------------------------------------------
# Compute bucket for every pixel
# ------------------------------------------------------------

for iy, y in enumerate(ys):

    for ix, x in enumerate(xs):

        codes = []

        for _, row in hashes.iterrows():

            projection = row["ax"]*x + row["ay"]*y

            code = int(np.floor((projection + row["offset"])/row["width"]))

            codes.append(code)

        bucket = bucket_to_string(codes)

        heat[iy, ix] = visits.get(bucket, 0)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 8))

heat[heat == 0] = 0.5

im = ax.imshow(
    heat,
    origin="lower",
    extent=[xmin, xmax, ymin, ymax],
    cmap="RdYlGn_r",
    norm=LogNorm(vmin=1, vmax=max(1, heat.max())),
    alpha=0.45,
    interpolation="nearest",
)

draw_hash_functions(ax, hashes, xmin, xmax, ymin, ymax)

plt.colorbar(im, label="Bucket visits")

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

ax.scatter(
    dataset["x"],
    dataset["y"],
    s=15,
    color="black",
    label="Dataset",
)

# ------------------------------------------------------------
# Query
# ------------------------------------------------------------

qx = query["x"][0]
qy = query["y"][0]

ax.scatter(
    qx,
    qy,
    color="blue",
    s=80,
    label="Query",
)

# ------------------------------------------------------------
# R sphere
# ------------------------------------------------------------

circle_R = Circle(
    (qx, qy),
    query["R"][0],
    fill=False,
    color="green",
    linewidth=2,
    label="R",
)

ax.add_patch(circle_R)

# ------------------------------------------------------------
# cR sphere
# ------------------------------------------------------------

circle_cR = Circle(
    (qx, qy),
    query["cR"][0],
    fill=False,
    color="red",
    linewidth=2,
    linestyle="--",
    label="cR",
)

ax.add_patch(circle_cR)

# ------------------------------------------------------------
# Decorations
# ------------------------------------------------------------

ax.set_aspect("equal")

ax.set_xlabel("x")
ax.set_ylabel("y")

ax.set_title("Adaptive Entropy-Guided LSH")

ax.legend()

plt.tight_layout()
plt.show()

