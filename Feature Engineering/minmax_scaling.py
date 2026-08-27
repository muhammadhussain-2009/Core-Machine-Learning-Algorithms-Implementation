import numpy as np
def min_max_scaling(data):
    """
    Scale each column of the data matrix to the [0, 1] range.
    """
    # Write code here
    data = np.array(data)
    if not isinstance(data, np.ndarray):
        raise ValueError("Incorrect Array Type")
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0.0] = 1.0
    scaled_mat = (data - min_vals) / ranges
    scaled_mat[:, max_vals == min_vals] = 0.0
    return scaled_mat.tolist()
    pass 
