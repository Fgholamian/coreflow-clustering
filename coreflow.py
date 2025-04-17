import numpy as np
from sklearn.neighbors import NearestNeighbors

# CoreFlow clustering algorithm
# This function iteratively shifts points toward the average of their local neighbors

# Parameters
#neighbors = 15        # Number of nearest neighbors used to influence each point. Typical range: 10–30 depending on dataset size and cluster density.
#step_size = 0.1       # Movement fraction per iteration toward the local core. Smaller values = smoother but slower convergence (e.g., 0.05–0.2)
#max_loops = 30        # Maximum number of iterations to run. Useful range: 20–50 for medium-sized datasets.
#tolerance = 1e-4      # Minimum movement threshold for convergence. Lower values mean more precise convergence, e.g., 1e-4 to 1e-3

def coreflow_move_to_cores(data_points, neighbors, step_size, max_loops, tolerance):
    updated_points = data_points.copy()
    num_points = len(updated_points)

    # Find neighbors once and reuse
    neighbor_model = NearestNeighbors(n_neighbors=neighbors).fit(updated_points)
    _, neighbor_indices = neighbor_model.kneighbors(updated_points)

    for loop in range(max_loops):
        shift_vectors = np.zeros_like(updated_points)

        for idx in range(num_points):
            local_neighbors = updated_points[neighbor_indices[idx]]
            avg_direction = np.mean(local_neighbors - updated_points[idx], axis=0)
            shift_vectors[idx] = avg_direction

        updated_points += step_size * shift_vectors

        # Check for convergence
        if np.max(np.linalg.norm(shift_vectors, axis=1)) < tolerance:
            break

    return updated_points


