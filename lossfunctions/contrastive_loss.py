# Implement Contrastive Loss for Siamese pairs. Given embeddings ai, bi ∈ ℝd and labels yi ∈ {0,1} where y=1 means "similar/positive" and y=0 means "dissimilar/negative":
# Loss= yid^2 + (1-y)max(0, m-d)^2 where d is the distance 

import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction='mean'):
    # Ensure inputs are numpy arrays
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    y = np.array(y).flatten()
    
    # Calculate Euclidean distance (d_i)
    # axis=1 computes norm across features (columns)
    d = np.linalg.norm(a - b, axis=1)
    
    # Calculate Contrastive Loss: y * d^2 + (1 - y) * max(0, margin - d)^2
    loss = y * np.square(d) + (1 - y) * np.square(np.maximum(0, margin - d))
    
    # Apply reduction
    if reduction == 'mean':
        return float(np.mean(loss))
    elif reduction == 'sum':
        return float(np.sum(loss))
    else:
        return loss
    pass
