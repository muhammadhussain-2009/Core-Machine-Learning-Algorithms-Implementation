# Calculate Entropy for a Node 
# Entropy is a fundamental concept from information theory that measures the amount of uncertainty or randomness in a dataset. 
# In decision trees, it's used as a splitting criterion to build trees that maximize information gain.

import numpy as np

def entropy_node(y):
    values, counts = np.unique(y, return_counts= True)
    p = counts/ len(y)
    sum = 0
    for i in p:
        sum += (i * np.log2(i))
    return float(sum * (-1))
    pass
