import numpy as np
from sklearn.utils import resample
from design_matrix import create_design_matrix
from linear_model import LinearModel

def calculate_stats_with_bootstrap(x_train, x_test, y_train, y_test, z_train, z_test, n_bootstraps, degree, linear_model : LinearModel): 
    """
    Calculates MSE, bias and variance from given data and model with bootstrap.

    Parameters
    ----------
    x_train : np.array
        x_train data from train_test_split
    x_test : np.array
        x_test data from train_test_split
    y_train : np.array
        y_train data from train_test_split
    y_test : np.array
        y_test data from train_test_split
    z_train : np.array
        z_train data from train_test_split
    z_test : np.array
        z_test data from train_test_split
    n_bootstraps: int
        Number of bootstraps passes
    degree: int
        The current wanted polynomial degree used to create the design matrix
    linear_model: LinearModel
        Linear model object used during the bootstrap to fit and predict values

    Returns
    -------
        degree: int
            The current degree used to create the design matrices
        error: float
            The mean square error calculated during the bootstrap
        bias: float
            The bias calculated during the bootstrap
        variance: float
            The variance calculated during the bootstrap
    """

    # Matrix holding the predictions from predict
    z_pred = np.empty((len(z_test), n_bootstraps))

    # Creating design matrix for test data for the given degree
    X_test = create_design_matrix(x_test,y_test, degree)
    
    # Creating design matrix for the train data for the given degree
    X_train = create_design_matrix(x_train,y_train, degree)

    # Running bootstrapping on the data
    for i in range(n_bootstraps):
        X_, z_ = resample(X_train, z_train)
        linear_model.fit(X_, z_)
        z_pred[:, i] = linear_model.predict(X_test).ravel()

    # Returning the error, bias and variance for the predictions  bootstrap.
    error = np.mean( np.mean((z_test - z_pred)**2, axis=1, keepdims=True) )
    bias = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
    variance = np.mean( np.var(z_pred, axis=1, keepdims=True) )

    print('Polynomial degree:', degree)
    print('Error:', error)
    print('Bias^2:', bias)
    print('Var:', variance)
    print('{} >= {} + {} = {}'.format(error, bias, variance, bias+variance))

    return degree, error, bias, variance