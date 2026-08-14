# Linear Layer transforms an input by multiplying with a weight matrix and adding a bias vector. This operation is also called a dense layer or an affine transformation.

import numpy as np
def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    # Write code here\
    X_arr = np.array(X, dtype=float)
    W_arr = np.array(W, dtype=float)
    b_arr = np.array(b, dtype=float)
    Y = np.dot(X_arr, W_arr) + b_arr
    return Y.tolist()
    pass
