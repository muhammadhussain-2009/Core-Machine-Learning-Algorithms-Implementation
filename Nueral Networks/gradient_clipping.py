# Implement global norm gradient clipping to prevent exploding gradients in deep networks. Scale gradients proportionally when their norm exceeds a threshold.

import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    g = np.asarray(g)
    if max_norm <= 0:
        return g.copy()
    total_norm = np.linalg.norm(g)
    if total_norm == 0:
        return g.copy()
    if total_norm > max_norm:
        return g * (max_norm / total_norm)
    return g.copy()
    pass
