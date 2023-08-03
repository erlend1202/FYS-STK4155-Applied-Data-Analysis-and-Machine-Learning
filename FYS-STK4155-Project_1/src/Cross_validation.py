import numpy as np
from sklearn.model_selection import KFold
from mean_square_error import MSE
from linear_model import LinearModel

def calculate_stats_with_crossvalidation(X, z, k, linear_model : LinearModel): 
    """
    Calculated and returns error (MSE) with crossvalidation (k fold).

    Parameters
    ----------

    X: np.array
        Current design matrix created with x, y values with a certain polynomial degree
    z: np.array
        Current z values.
    k: int
        Number of folds in the k fold algorithm
    linear_model: LinearModel
        Linear model object used during the crossvalidation to fit and predict values

    Returns
    -------
    error: float
        Mean square error calculated with crossvalidation and the given design matrix.

    """

    k_fold = KFold(n_splits=k)

    current_error_values = []

    # Predicting and calculating MSE for all variations test/train folds
    for train_inds, test_inds in k_fold.split(X):

        X_train = X[train_inds]
        X_test = X[test_inds]
        z_train = z[train_inds]
        z_test = z[test_inds]
        
        linear_model.fit(X_train, z_train)
        y_tilde = linear_model.predict(X_test).ravel()

        error = np.mean(MSE(z_test, y_tilde))
        current_error_values.append(error)
    
    error = np.mean(current_error_values)

    return error