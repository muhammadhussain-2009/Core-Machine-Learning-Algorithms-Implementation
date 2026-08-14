# He (Kaiming) initialization is designed for networks using ReLU activations. 
# Since ReLU zeros out negative values (roughly half the distribution), He initialization uses a larger variance than Xavier to compensate, depending only on the fan-in.

# He uniform bound = sqrt(6/fan_in)

import numpy as np
def he_initialization(W, fan_in):
    """
    Scale raw weights to He uniform initialization.
    """
    # Write code here
    W= np.array(W)
    L = np.sqrt(6/fan_in)
    new_w = (W * 2 * L) - L
    return new_w
    pass 
