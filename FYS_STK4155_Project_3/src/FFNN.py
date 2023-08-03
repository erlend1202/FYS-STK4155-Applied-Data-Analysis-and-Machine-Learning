from cmath import isnan, nan
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt 
from mean_square_error import MSE
from activation_functions import * 
from design_matrix import *

class FeedForwardNeuralNetwork:
    """
        A class representing a feed forward neural network. Can be used for both regression and classification problems. 
        The class supports for a arbritary amount and size of hidden layers and custom activation functions where the default
        activation function is the sigmoid function.
    """
    def __init__(self, X, Y, layers, n_categories = 1, batch_size = 5, eta = 0.1, lmbda = 0.0, epochs = 10, func=sigmoid, problem="regression"):
        """
        Setting up the neural network variables and initializing weights and biases in the network.

        Parameters
        ----------
        X: np.ndarray
            The current design matrix to be used. Can be in any polynomial degree.
        Y: np.array
            The y vector used to perform the fitting.
        layers: np.array
            A list of all the hidden layers in the network. Elements in the list are a numerical value representing the number of node in the layer. 
        n_categories: int
            The number of output layers in the network.
        batch_size: int
            The batch size the network should use during the training process.
        eta: float
            The eta hyperparameter the network should use.
        lmbda: float
            The lambda hyperparameter the network should use.
        epochs: int
            The number of epochs to use when training the network.
        func: Function
            The activation function the network should use.
        problem: string
            The type of problem. Either regression or classification in this case. 
        """
        self.X = X
        self.Y = Y
        self.func = func
        self.problem = problem
        self.n_inputs = X.shape[0] # Samples
        self.n_features = X.shape[1]
        self.n_categories = n_categories
        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbda = lmbda
        self.layers = layers 
        self.num_layers = len(layers)

        # For eta
        self.t0, self.t1 = 5, 50

        # Creating biases and weights with initial values
        self.create_biases_and_weights()

    def learning_schedule(self, t):
        return self.t0 / (t + self.t1)

    def create_biases_and_weights(self):
        """
            This function is setting up all the layers with weights and biases.
        """
        self.weights = []
        self.bias = []
        for i in range(self.num_layers):
            if i==0:
                w = np.random.randn(self.n_features, self.layers[i])
                b = np.zeros(self.layers[i]) + 0.01
            else:
                w = np.random.randn(self.layers[i-1], self.layers[i])
                b = np.zeros(self.layers[i]) + 0.01
            self.weights.append(w)
            self.bias.append(b)
        w = np.random.randn(self.layers[i], self.n_categories)
        b = np.zeros(self.n_categories) + 0.01
        self.weights.append(w)
        self.bias.append(b)

        #Changing from list to array 
        self.weights = np.array(self.weights)
        self.bias = np.array(self.bias)

    def feed_forward(self): 
        """
            The feed forward stage in the neural network. 
        """
        self.z = []
        self.a = []

        for i in range(self.num_layers):
            if i == 0:
                z = np.matmul(self.current_X_data, self.weights[i]) + self.bias[i]
                a = self.func(z)
            else:
                z = np.matmul(self.a[i-1], self.weights[i]) + self.bias[i]
                a = self.func(z)

            self.z.append(z)
            self.a.append(a)

        z = np.matmul(self.a[-1], self.weights[-1]) + self.bias[-1]
        self.z.append(z)
        self.a.append(z)

        if self.problem == "regression":
            self.probabilities = z

        #Assume its classification otherwise
        else:
            exp_term = np.exp(z)
            self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)


    def feed_forward_out(self, X):
        """
            The feed forward stage in the neural network with returning out value(s) of the network.

            Returns
            -------
            np.array
                Returns the output vaslues from the last layer in the network.
        """
        z_list = []
        a_list = []

        for i in range(self.num_layers):
            if i == 0:
                z = np.matmul(X, self.weights[i]) + self.bias[i]
                a = self.func(z)
            else:
                z = np.matmul(a_list[i-1], self.weights[i]) + self.bias[i]
                a = self.func(z)

            z_list.append(z)
            a_list.append(a)
            
        z = np.matmul(a_list[-1], self.weights[-1]) + self.bias[-1]
        
        if self.problem == "regression":
            return z

        #Asume its classification otherwise
        else:
            exp_term = np.exp(z)
            return exp_term / np.sum(exp_term, axis=1, keepdims=True)
    
    def backpropagation(self):
        """
            The backpropagation stage in the neural network.
        """
        error1 = self.probabilities - self.current_Y_data
        errors = [error1]
        self.w_grads = []
        self.bias_grads = []

        for i in range(self.num_layers):
            if i == 0:
                if self.func == sigmoid:
                    error = np.matmul(errors[i], self.weights[self.num_layers].T) * self.a[self.num_layers-1] * (1-self.a[self.num_layers-1])
                elif self.func == relu:
                    error = np.matmul(errors[i], self.weights[self.num_layers].T) * delta_relu(self.z[self.num_layers-1])
                elif self.func == leaky_relu:
                    error = np.matmul(errors[i], self.weights[self.num_layers].T) * delta_leaky_relu(self.z[self.num_layers-1])

            else:
                if self.func == sigmoid:
                    error = np.matmul(errors[i], self.weights[self.num_layers-i].T) * self.a[self.num_layers-i-1] * (1-self.a[self.num_layers-i-1])
                elif self.func == relu:
                    error = np.matmul(errors[i], self.weights[self.num_layers-i].T) * delta_relu(self.z[self.num_layers-1-i])
                elif self.func == leaky_relu:
                    error = np.matmul(errors[i], self.weights[self.num_layers-i].T) * delta_leaky_relu(self.z[self.num_layers-1-i])


            dw = np.matmul(self.a[self.num_layers-1-i].T, errors[i])
            db = np.sum(errors[i], axis=0)

            errors.append(error)
            self.w_grads.append(dw)
            self.bias_grads.append(db)

        dw = np.matmul(self.current_X_data.T, errors[self.num_layers])
        db = np.sum(errors[self.num_layers], axis=0)

        self.w_grads.append(dw)
        self.bias_grads.append(db)


        if self.lmbda > 0:
            for i in range(self.num_layers+1):
                self.w_grads[i] += self.w_grads[i]*self.lmbda

        for i in range(self.num_layers+1):
            self.weights[i] -= self.eta * self.w_grads[self.num_layers-i]
            self.bias[i] -= self.eta * self.bias_grads[self.num_layers-i]

    def predict(self, X):
        """
            Used for making a prediction with the neural network. The network needs to previously be trained. Used for classification.

            Parameters
            ----------
            X: np.ndarray
                Values to feed into the network to make the prediction.
            
            Returns
            -------
            int
                Returns the index of the maximum values of the network output. Used for classification.
        """
        probabilities = self.feed_forward_out(X)
        return np.argmax(probabilities, axis=1)
    
    def predict_probabilities(self, X):
        """
            Used for making a prediction with the neural network. The network needs to previously be trained. Used for regression.

            Parameters
            ----------
            X: np.ndarray
                Values to feed into the network to make the prediction.
            
            Returns
            -------
            np.array
                Returns the probabilities from the network output. 
        """
        probabilities = self.feed_forward_out(X)
        return probabilities
    
    def train(self):
        """
            Used for training the network with the data provided during the initialization of the class. Trains the network for the provided 
            amount of epochs.
        """
        data_indices = np.arange(self.n_inputs)

        for i in range(self.epochs):
            for j in range(self.iterations):
                self.eta = self.learning_schedule(i * self.iterations + j)

                chosen_datapoints = np.random.choice(data_indices, size=self.batch_size, replace=False)

                self.current_X_data = self.X[chosen_datapoints]
                self.current_Y_data = self.Y[chosen_datapoints]

                self.feed_forward()
                self.backpropagation()
