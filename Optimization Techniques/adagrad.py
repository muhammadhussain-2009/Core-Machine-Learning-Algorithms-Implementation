# Implement one AdaGrad update. First accumulate the elementwise squared gradient, then update each parameter

import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    """
    # Write code here
    w, g, G = np.asarray(w), np.asarray(g), np.asarray(G)
    new_G= G + np.power(g,2)
    new_w = w - (lr/np.sqrt(new_G+eps))*g
    return new_w, new_G
    pass
