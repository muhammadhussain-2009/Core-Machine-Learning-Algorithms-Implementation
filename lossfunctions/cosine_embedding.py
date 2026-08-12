# Cosine embedding loss measures whether two vectors are similar or dissimilar based on a label.
#  It is commonly used in metric learning and siamese networks to learn embeddings where similar items are close and dissimilar items are far apart in cosine space.
# Given two vectors, a label (+1 for similar, -1 for dissimilar), and a margin, compute the cosine embedding loss.

import numpy as np 
import math 

def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    # Write code here
    norm_x1 = np.linalg.norm(x1, ord=2)
    norm_x2 = np.linalg.norm(x2, ord=2)
    cosine_similarity = np.dot(x1,x2)/np.dot(norm_x1, norm_x2)
    if label == 1:
        L= 1 - cosine_similarity
    if label ==  -1:
        L = np.maximum(0, cosine_similarity-margin)
    return float(L)
    pass 
