# Implement a majority class classifier that predicts the most frequent label in the training data for all test samples.

import numpy as np

def majority_classifier(y_train, X_test):
    #write code here
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    num_test_samples= X_test.shape[0] if X_test.ndim > 0 else 0.0
    if num_test_samples == 0:
        return np.array([], dtype=int)
    if y_train.size == 0:
        return np.zeros(num_test_samples, dtype=int)
    classes, counts = np.unique(y_train, return_counts=True)
    majority_class = classes[np.argmax(counts)]
    return np.full(num_test_samples, majority_class, dtype=int)
    pass
