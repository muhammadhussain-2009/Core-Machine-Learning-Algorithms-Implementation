# Implement Triplet Loss for embedding ranking. This loss ensures that an anchor embedding is closer to a positive example than to a negative example by a margin.
# Triplet Loss: max(0, d(a,p), d(a,n) + m ) where d is the distance, m is the margin, a is the anchor embedding, p is the positive example and n is the negative example 

import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss for embedding ranking.
    """
    # Write code here
    a, p, n = np.array(anchor), np.array(positive), np.array(negative)
    dist_ap = np.sum(np.square(a - p), axis=-1)
    dist_an = np.sum(np.square(a - n), axis=-1)
    loss = np.maximum(0, dist_ap - dist_an + margin)
    return float(np.mean(loss))
    pass
