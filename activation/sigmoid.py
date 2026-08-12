# Vectorised Sigmoid Activation Function
# sigmoid(x) = 1/1+e^-x

import numpy as np

def sigmoid(x):
    arr= np.array(x)
    return 1/(1+ np.exp(-arr))
    pass 
