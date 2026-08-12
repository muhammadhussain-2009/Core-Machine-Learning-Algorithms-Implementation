# Label smoothing is a regularization technique that prevents a model from becoming overconfident. 
# Instead of using hard one-hot targets (1 for the correct class, 0 for all others), it softens the target distribution by redistributing a small fraction of probability mass to the incorrect classes.
# Given a predicted probability distribution, a target class index, and a smoothing parameter epsilon, compute the cross-entropy loss with smoothed labels.

import numpy as np
def label_smoothing_loss(predictions, target, epsilon):
    """
    Compute cross-entropy loss with label smoothing.
    """
    # Write code here
    predictions = np.array(predictions)
    K = predictions.shape[0]
    q = np.full(K, epsilon / K)
    q[target] = (1.0 - epsilon) + (epsilon / K)
    p = np.clip(predictions, 1e-15, 1.0)
    loss = -np.sum(q * np.log(p))
    return float(loss)
    pass 
