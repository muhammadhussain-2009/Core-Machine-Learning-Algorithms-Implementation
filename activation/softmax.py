# Softmax function, which converts raw scores (logits) into probabilities that sum to 1.

import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims = True))
    return exp_x/np.sum(exp_x, axis = -1, keepdims= True)
    pass
