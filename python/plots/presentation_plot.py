import matplotlib as mpl
mpl.use('WebAgg')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS

# --- 1. Create the Dataset: Two Blobs and a Sparse Bridge ---

# 1.1. Cluster 1 (Dense)
X_cluster1, y_c1 = make_blobs(n_samples=200, centers=[[0, 0]], 
                              cluster_std=0.5, random_state=42)

# 1.2. Cluster 2 (Dense)
X_cluster2, y_c2 = make_blobs(n_samples=200, centers=[[5, 5]], 
                              cluster_std=0.4, random_state=42)
y_c2 += 1

# 1.3. Sparse Bridge (Noise)
# Create a sparse, linear connection between the two clusters
N_bridge = 30
np.random.seed(42)
t = np.linspace(0, 1, N_bridge).reshape(-1, 1)
# Linear interpolation between [0, 0] and [5, 5]
X_bridge = t * np.array([5, 5]) + (1 - t) * np.array([0, 0])
# Add a lot of noise to ensure low density
X_bridge += np.random.randn(N_bridge, 2) * 0.5 
y_bridge = np.full(N_bridge, 2)

# Combine the datasets
X = np.vstack([X_cluster1, X_cluster2, X_bridge])
y_combined = np.concatenate([y_c1, y_c2, y_bridge])
N = X.shape[0]
MIN_SAMPLES = 2 # k parameter (MinPts) - Crucial for density definition

# --- 2. Calculate Mutual Reachability Distance (MRD) Matrix ---

# Step A: Calculate Core Distance
nn = NearestNeighbors(n_neighbors=MIN_SAMPLES, algorithm='auto').fit(X)
distances, indices = nn.kneighbors(X) 
core_distances = distances[:, MIN_SAMPLES - 1] 

# Step B: Calculate Euclidean Distance Matrix
euclidean_distances = squareform(pdist(X, metric='euclidean'))

# Step C: Calculate MRD Matrix: MRD(a, b) = max(core(a), core(b), dist(a, b))
core_max_distances = np.maximum.outer(core_distances, core_distances)
mrd_matrix = np.maximum(core_max_distances, euclidean_distances)


# --- 3. Perform Spatial Transformation using MDS ---

# 3.1. MRD-Based MDS (Transformed Space)
# Use the MRD matrix to find new coordinates that best preserve these distances
# Increased max_iter for convergence on the complex MRD metric
mds_mrd = MDS(n_components=2, dissimilarity='precomputed', random_state=42, 
              n_init=1, max_iter=5)
X_mds_mrd = mds_mrd.fit_transform(mrd_matrix)


# --- 4. Plotting the Comparison ---

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True, constrained_layout=True)
titles = [
    'Original Data',
    'Data Transformed with MRD'
]
datasets = [X, X_mds_mrd]
core_dist_color = core_distances 

# Get global min/max for consistent color scaling across plots
v_min, v_max = core_dist_color.min(), core_dist_color.max()

for i, ax in enumerate(axes):
    data = datasets[i]
    
    # Plotting where color represents the Core Distance (local density)
    scatter = ax.scatter(data[:, 0], data[:, 1], c=core_dist_color, 
                         cmap='viridis_r', s=40, alpha=0.9,
                         vmin=v_min, vmax=v_max) # Use viridis_r for dark = dense
    
    ax.set_title(titles[i], fontsize=14)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect('equal', adjustable='box') # Keep aspect ratio consistent

# Add a single color bar
cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), label=f'Core Distance (k={MIN_SAMPLES})', 
                    shrink=0.8)

# Add a text box to highlight the key effect

plt.savefig('hdbscan_mrd_transformation.png', dpi=300)