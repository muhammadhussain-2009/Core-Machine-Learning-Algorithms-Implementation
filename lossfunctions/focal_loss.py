# Focal Loss for binary classification. This loss function addresses class imbalance by down-weighting easy examples and focusing training on harder ones.
# Mathematical Formula: FL(p,y) = -[(1-p)**gamma * y * np.log(p) + p**gamma * (1-y) * np.log(1-p)]

import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """
    # Write code here
    p=np.array(p)
    y = np.array(y)
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    term_1 = (1-p)**gamma * y * np.log(p)
    term_2 = p**gamma * (1-y) * np.log(1-p)
    focal_loss = -(term_1 + term_2)
    return float(np.mean(focal_loss))
    pass
