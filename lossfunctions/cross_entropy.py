#Compute the average cross-entropy loss for multi-class classification.
#y_true contains the correct class labels, y_pred contains the predicted class labels 
#For each sample the loss is -log(p). The final cross entropy is the average of the logs. 

import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    y_true= np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    N= y_pred.shape[0]
    correct_class_probs = y_pred[np.arange(N), y_true] #np.arange is used for advanced indexing and to extract the probabilities for the correct classes.
    logs = np.log(correct_class_probs)
    loss = -np.mean(logs)
    return loss 
    pass

# Example Test Case: 
# Input: y_true = [0, 1], y_pred = [[0.9, 0.1], [0.3, 0.7]] Output: 0.231018
