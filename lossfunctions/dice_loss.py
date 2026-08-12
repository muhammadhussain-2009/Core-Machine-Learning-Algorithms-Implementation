# Dice Loss for segmentation tasks. This loss measures overlap between prediction and ground truth masks, widely used in image segmentation where pixel imbalance is common.

import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    # Write code here
    p = np.array(p).flatten()
    y = np.array(y).flatten()
    intersection = np.sum(p * y)
    sum_p = np.sum(p)
    sum_y = np.sum(y)
    dice_coeff = (2.0 * intersection + eps) / (sum_p + sum_y + eps)
    return float(1.0 - dice_coeff)
    pass
