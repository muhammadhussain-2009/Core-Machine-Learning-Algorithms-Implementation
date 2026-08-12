#Compute the Hinge Loss or binary SVM with labels y ∈ {-1,+1} and real-valued scores s. For each example:
# Mathematical formula: hinge loss = max(0,m-y*s) where m is the margin 

import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    """
    y_true: 1D array of {-1,+1}
    y_score: 1D array of real scores, same shape as y_true
    reduction: "mean" or "sum"
    Return: float
    """
    # Write code here
    y_true = np.asarray(y_true) 
    y_score = np.asarray(y_score)
    hinge_loss= np.maximum(0,margin-(y_true*y_score))
    if reduction == "mean":
        hinge_loss= hinge_loss.mean()
    if reduction == "sum":
        hinge_loss= hinge_loss.sum()
    return float(hinge_loss)
    pass

#Example Test Case:
Input:
y_true = [ 1, 1, -1]
y_score = [ 2, 0, 0]
Output:
( max(0,1-1*2)=0, max(0,1-1*0)=1, max(0,1-(-1*0))=1 )
→ mean = (0+1+1)/3 = 0.66666667
