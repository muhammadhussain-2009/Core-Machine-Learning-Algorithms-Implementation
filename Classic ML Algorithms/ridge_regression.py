# Ridge regression (L2 regularization) adds a penalty term to the ordinary least squares objective to prevent overfitting. 
# The regularization term shrinks the weights toward zero, which is especially useful when features are correlated or when the number of features is large relative to the number of samples.
# formula: W = (XTX + (lambdaI))^-1 * XT * y

import numpy as np 
def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    # Write code here
    X= np.array(X)
    y = np.array(y)
    n_samples, n_features = X.shape 
    part1 = np.dot(X.T, X)
    I = np.eye(n_features)
    regularized_term = part1 + (lam * I)
    weights = np.linalg.inv(regularized_term).dot(X.T).dot(y)
    return weights.tolist()
    
