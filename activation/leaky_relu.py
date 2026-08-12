# Leaky ReLU is defined as:
# f(x) = x if x >= 0 and f(x) = ax when x < 0

import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    x= np.asarray(x)
    leaky_relu = np.where(x>0,x,x*alpha)
    return leaky_relu
    pass
