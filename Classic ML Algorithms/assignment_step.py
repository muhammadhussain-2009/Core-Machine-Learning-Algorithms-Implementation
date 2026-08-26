#  The assignment step assigns each data point to the nearest centroid based on squared Euclidean distance.
# Given a list of data points and a list of current centroid positions, assign each point to the nearest centroid.

import numpy as np
def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    # Write code here
    points = np.array(points, dtype=float)
    centroids = np.array(centroids, dtype=float)
    dists = np.sum((points[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
    assignments = np.argmin(dists, axis=1)
    return assignments.tolist()
    pass 
