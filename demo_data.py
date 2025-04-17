import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from coreflow import coreflow_move_to_cores
from sklearn.cluster import DBSCAN

# synthetic data 4 cluster blobs
input_data, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.2, random_state=42)

core_positions = coreflow_move_to_cores(input_data, neighbors=15, step_size=0.1, max_loops=30, tolerance = 1e-4)

#predicted_labels = AgglomerativeClustering(n_clusters=4).fit_predict(core_positions)


predicted_labels = DBSCAN(eps=0.6, min_samples=5).fit_predict(core_positions)

fig, axis = plt.subplots(1, 2, figsize=(12, 5))
axis[0].scatter(input_data[:, 0], input_data[:, 1], c='gray', alpha=0.6)
axis[0].set_title("Original Data")
axis[1].scatter(input_data[:, 0], input_data[:, 1], c=predicted_labels, cmap='tab10', alpha=0.8)
axis[1].set_title("CoreFlow Clustering Result")
plt.tight_layout()
plt.show()
