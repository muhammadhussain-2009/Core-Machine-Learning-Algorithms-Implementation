#  KL Divergence (Kullback-Leibler divergence) to measure how one probability distribution differs from another.
# p: array first probability distribution Shape (N,)
# q: array second probability distribution Shape (N,) 

import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    # Write code here
    p = np.array(p)
    q = np.array(q)
    p_stable = p + eps
    q_stable = q + eps
    kl_divergence = np.sum(p*np.log(p_stable/q_stable))
    return kl_divergence
    pass
