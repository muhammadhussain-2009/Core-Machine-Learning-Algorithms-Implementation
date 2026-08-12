# the Tanh (Hyperbolic Tangent) activation function. Tanh squashes values to the range (-1, 1) and is symmetric around zero.
# tanh = (e^x = e^-x)/ (e^x + e^-x)

import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    x=np.asarray(x)
    numerator= (np.exp(x))-(np.exp(-x))
    denominator = (np.exp(x))+(np.exp(-x))
    tanh = numerator/denominator
    return tanh
    pass
