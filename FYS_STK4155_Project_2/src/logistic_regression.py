# Using Autograd to calculate gradients for OLS
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from activation_functions import * 
from to_categorical import *
from accuracy_score import *

class NumpyLogReg:
    """
        A class containing a fit and predict function for a logistic regression model.
    """

    def fit(self, X, y, eta = 0.1, epochs=10, M=5, lmbda=0.1):
        """
            Fit function for training the model.

            Parameters
            ----------
            X: np.ndarray
                Design matrix which represents the input data for training the model.
            y: np.array
                y values which represents the target values during the training process of the model.
            eta: float
                The eta hyperparameter the model should use.
            lmbda: float
                The lambda hyperparameter the model should use.
            epochs: int
                The number of epochs to use when training the model.
            M: int
                The M hyperparameter the model should use when training. 
        """
        (k, n) = X.shape
        self.weights = weights = np.zeros(n)

        m = int(k/M)
             
        change = 0
        
        #for changing learning rate
        self.t0, self.t1 = 5, 50

        for iter in range(epochs):
            for i in range(m):
                random_index = np.random.randint(m)*M
                xi = X[random_index:random_index+M]
                yi = y[random_index:random_index+M]
                #Changing eta over time
                eta = self.learning_schedule(iter*m+i)

                gradients = eta / k *  xi.T @ (self.forward(xi) - yi) + lmbda*change 

                change = gradients
                weights -= change

        self.weights = weights
            
    def learning_schedule(self, t):
        """
            Calculates the learning schedule
        """
        return self.t0/(t+self.t1)

    def forward(self, X):
        """
            Forward pass in the logistic regression algorithm. Using the sigmoid function as activation
            function.

            Parameters
            ----------
            X: np.ndarray
                Input values in the forward pass.
            
            Returns
            -------
            float 
                Values multiplied with the weights and passed though the activation function.

        """
        return sigmoid(X @ self.weights)
    
    def score(self, x):
        """
            Returns the result from a forward pass with the x data. 

            Parameters
            ----------
            x: np.ndarray
                Input values in the forward pass

            Returns
            -------
                Returns the result of a forward pass in the model.
        """
        score = self.forward(x)
        return score
    
    def predict(self, x, threshold=0.5):
        """
            Returning a prediction with using the already trained model.

            Parameters
            ----------
            x: np.ndarray 
                Input values to the model to create a prediction from. 
            
            Returns
            -------
            int 
                Returns true if score is larger than the threshold. Otherwise, returning false.
        """
        z = x.copy()
        score = self.forward(z)
        return (score>threshold).astype('int')
    
    def loss(self, y, y_hat):
        """
            Calculates loss from values and predicted values.

            Parameters
            ----------
            y: np.array 
                Real values
            y_hat: np.array 
                Predicted values

            Returns
            -------
            float 
                Returns the loss value.
        """
        return -np.mean(y*(np.log(y_hat)) - (1-y)*np.log(1-y_hat))