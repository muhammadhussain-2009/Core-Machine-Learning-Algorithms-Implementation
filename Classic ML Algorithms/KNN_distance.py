# Given training data, test data, and k, compute pairwise Euclidean distances and return the k nearest neighbor indices for each test point.

import numpy as np 

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    # Write code here
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    if X_train.ndim == 1: X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1: X_test = X_test.reshape(-1, 1)
    n_train = X_train.shape[0]
    actual_k = min(k, n_train)
    diff = X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))
    partitioned_indices = np.argpartition(distances, actual_k - 1, axis=1)[:, :actual_k]
    row_indices = np.arange(X_test.shape[0])[:, np.newaxis]
    sorted_k_indices = partitioned_indices[
        row_indices, 
        np.argsort(distances[row_indices, partitioned_indices], axis=1)
    ]
    if k > n_train:
        res = np.full((X_test.shape[0], k), -1, dtype=int)
        res[:, :n_train] = sorted_k_indices
        return res 
    return sorted_k_indices
    pass
