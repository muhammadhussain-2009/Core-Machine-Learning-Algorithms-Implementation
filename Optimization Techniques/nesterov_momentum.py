# Perform one Nesterov momentum update using a gradient already evaluated at the look-ahead position.

import numpy as np

def nesterov_momentum_step(w, v, grad, lr=0.01, momentum=0.9):
    """
    Perform one Nesterov Momentum update step.
    """
    # Write code here
    weights = np.array(w, dtype=float)
    velocity = np.array(v, dtype=float)
    grad = np.array(grad, dtype=float)
    updated_v = momentum * velocity + lr * grad
    updated_w = weights - updated_v
    return updated_w, updated_v
    pass
