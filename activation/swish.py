#  Swish is a smooth, learnable activation function that often improves performance over ReLU due to smoother gradient flow.
#  Swish = x * sigmoid(x)

import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    try:
        x_arr = np.asarray(x, dtype=float)
    except Exception as e:
        raise ValueError("Input must be numeric or array-like.") from e
    sigmoid = 1 / (1 + np.exp(-x_arr))
    return x_arr * sigmoid
    pass
