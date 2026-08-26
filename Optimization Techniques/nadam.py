# Nesterov + Adam update without bias correction

import numpy as np

def nadam_step(w, m, v, grad, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Perform one Nadam update step.
    """
    w = np.array(w)
    m = np.array(m)
    v = np.array(v)
    grad = np.array(grad)
    m_new = beta1 * m + (1 - beta1) * grad
    v_new = beta2 * v + (1 - beta2) * (grad**2)
    numerator = beta1 * m_new + (1 - beta1) * grad
    denominator = np.sqrt(v_new) + eps
    w_new = w - lr * (numerator / denominator)
    
    return w_new, m_new, v_new
    pass
