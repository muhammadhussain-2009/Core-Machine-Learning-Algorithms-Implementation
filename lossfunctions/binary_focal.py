# Binary focal loss addresses the class imbalance problem in binary classification by down-weighting the loss contribution from easy (well-classified) examples
# This allows the model to focus training on hard, misclassified examples. It was introduced in the RetinaNet paper for object detection.

import numpy as np
def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    """
    predictions, targets = np.array(predictions), np.array(targets)
    p_t = np.where(targets == 1, predictions, 1 - predictions)
    epsilon = 1e-12
    fl = -alpha * (1 - p_t)**gamma * np.log(p_t)
    return float(np.mean(fl))
    pass
