# A Random Forest makes predictions by aggregating the outputs of multiple decision trees. 
# For classification, each tree votes for a class and the final prediction is the class with the most votes

import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    # Write code here
    predictions = np.array(predictions)
    T, N = predictions.shape
    majority_votes = []
    for i in range(N):
        sample_predictions = predictions[:, i]
        unique_classes, counts = np.unique(sample_predictions, return_counts=True)
        max_count = np.max(counts)
        tied_classes = unique_classes[counts == max_count]
        majority_votes.append(int(np.min(tied_classes)))
    return majority_votes
    pass 
