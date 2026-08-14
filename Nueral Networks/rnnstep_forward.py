# Implement a single RNN step for a basic tanh RNN cell. At each time step, the RNN takes the current input and previous hidden state to compute the new hidden state.

import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    """
    # Write code here
    h_t = np.tanh((x_t @ Wx)+(h_prev @ Wh)+b)
    h_t = np.asarray(h_t)
    return h_t
    pass
