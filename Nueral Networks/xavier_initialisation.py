# Xavier (Glorot) initialization sets initial weights to maintain roughly the same variance of activations and gradients across layers. 
# This prevents the vanishing or exploding gradient problem in networks using sigmoid or tanh activations.

import numpy as np 
def xavier_initialization(W, fan_in, fan_out):
    """
    Scale raw weights to Xavier uniform initialization.
    """
    # Write code here
    W= np.array(W)
    L = np.sqrt(6/(fan_in+fan_out))
    new_w = (W * 2*L) - L 
    return new_w 
    pass 
