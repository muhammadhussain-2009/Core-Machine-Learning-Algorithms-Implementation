# Given a list of data points, their cluster assignments, and the number of clusters k, compute the new centroid positions.

import numpy as np
def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    # Write code here
    points, assignments = np.array(points), np.array(assignments)
    n_features = points.shape[1]
    centroids = np.zeros((k, n_features))
    for j in range(k):
        cluster_mask = (assignments == j)
        if np.any(cluster_mask):
            centroids[j] = np.mean(points[cluster_mask], axis=0)
        else:
            centroids[j] = 0.0  
    return centroids.tolist()
    pass 
