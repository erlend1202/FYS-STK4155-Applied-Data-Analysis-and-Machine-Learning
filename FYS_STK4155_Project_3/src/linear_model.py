import numpy as np
from enum import Enum
from sklearn.linear_model import Lasso

class LinearModelType(Enum):
    OLS = 1
    RIDGE = 2
    LASSO = 3

class LinearModel: 
    """
    A class containing fit and predict functions for OLS, ridge and lasso regression.
    """

    def __init__(self, linear_model_type: LinearModelType):
        """
        Setting up the linear model for the given linear model type.
        Parameters
        ----------
        linear_model_type: LinearModelType
            The current linear model type. Either OLS, RIDGE or LASSO.
        """
        self.current_linear_model_type = linear_model_type
        self.lmda = 1
        self.lasso_model = Lasso(alpha = self.lmda, fit_intercept=False, max_iter=1000, tol=0.1, normalize=True)
        self.betas = None

    def fit(self, X, y):
        """
        Fit function for the model.
        Parameters
        ----------
        X: np.array
            The current design matrix to be used. Can be in any polynomial degree.
        y: np.array
            The y vector used to perform the fitting. 1D array
        """
        if self.current_linear_model_type == LinearModelType.OLS: 
            self.ordinary_least_squares_fit(X, y)

        elif self.current_linear_model_type == LinearModelType.RIDGE: 
            self.ridge_fit(X, y)

        elif self.current_linear_model_type == LinearModelType.LASSO: 
            self.lasso_model.fit(X, y)

    def predict(self, pred_data):
        """
            Can be used after the fit function. Returns a prediction based on the previous fit and 
            given data.
            Parameters
            ----------
            pred_data: np.array
                Test-data used to predict based on the previous fit.
            
            Returns
            -------
            np.array
                Returns the prediction made based on pred_data and the previous fit.
        """

        if self.current_linear_model_type == LinearModelType.OLS: 
            return pred_data @ self.betas

        elif self.current_linear_model_type == LinearModelType.RIDGE: 
            return pred_data @ self.betas

        elif self.current_linear_model_type == LinearModelType.LASSO:
            return self.lasso_model.predict(pred_data)

    def ordinary_least_squares_fit(self, X, y): 
        self.betas =  np.linalg.pinv(X.T @ X) @ X.T @ y

    def ridge_fit(self, X, y): 
        I = np.eye(len(X[0]), len(X[0]))
        self.betas = np.linalg.pinv(X.T @ X + self.lmda * I) @ X.T @ y

    def set_lambda(self, lmda):
        """
            Setting the lamnda used in ridge regression. This is also used to set the alpha in lasso regression.
            Parameters
            __________
            lmda: float
                The current lamnda to later be used in ridge or as alpha in lasso regression
        """
        self.lmda = lmda
        self.lasso_model = Lasso(alpha = self.lmda, fit_intercept=False, max_iter=1000, tol=0.01, normalize=True)

    def set_linear_model_type(self, type: LinearModelType): 
        """ 
            Setting the current type of model to use.
            Parameters
            ----------
            type: LinearModelType
                The current type of model to use. Can be either set to OLS, RIDGE or lasso. 
        """
        self.current_linear_model_type = type