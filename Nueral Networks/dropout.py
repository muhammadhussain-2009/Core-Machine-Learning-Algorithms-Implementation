# Implement dropout for training. Randomly set elements to zero with probability p, then scale the remaining elements by 1/1-p to main the same expected value. 

import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x = np.asarray(x)
    if p < 0.0 and p>1.0:
        raise ValueError("Incorrect Probability")
    if rng is not None:
        value= rng.random(size=x.shape)
    else:
        value = np.random.random(size= x.shape)
    keep_mask = value < (1.0 - p)
    scale= 1.0/(1.0-p)
    dropout_pattern = keep_mask.astype(x.dtype) * scale
    out = x * dropout_pattern
    return (out, dropout_pattern)
    pass
