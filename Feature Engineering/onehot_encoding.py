import numpy as np

def one_hot(y, num_classes=None):
    """
    Convert integer labels y ∈ {0,...,K-1} into one-hot matrix of shape (N, K).
    """
    # Write code here
    y = np.array(y, dtype=int)
    if num_classes is None:
        num_classes = np.max(y) + 1 if y.size > 0 else 0
    one_hot = np.eye(num_classes)[y]
    return one_hot
    pass
