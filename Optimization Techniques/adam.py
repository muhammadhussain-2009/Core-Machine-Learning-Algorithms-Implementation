# Given current parameter(s), gradient(s), and running first/second moments, return the updated parameter(s) and updated moments.

import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    # Step 1: Update First Moment
    param = np.asarray(param)
    grad = np.asarray(grad)
    m = np.asarray(m)
    v = np.asarray(v)
    new_m = beta1 * m + (1 - beta1) * grad
    new_v = beta2 * v + (1 - beta2) * (grad**2)
    m_hat = new_m / (1 - beta1**t)
    v_hat = new_v / (1 - beta2**t)
    new_theta = param - lr * (m_hat / (np.sqrt(v_hat) + eps))
    return new_theta, new_m, new_v
    pass
