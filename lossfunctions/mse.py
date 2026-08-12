# Mean Squared Error 
# Mathematical Formula: 1/N * sum(y_pred[i]- y_true[i])**2 where y_pred is the array containing predicted values and y_true is the array containing true values.

import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    # Write code here
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must match.")
    mse = np.mean((y_true - y_pred) ** 2)
    return float(mse)
    pass
