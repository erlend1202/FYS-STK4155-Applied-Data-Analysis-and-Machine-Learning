import numpy as np

def sigmoid(x):
    """
        The sigmoid activation function.

        Parameters
        ---------
        x: float
            Input value to the sigmoid function.

        Returns
        -------
        float
            Returns output value from the sigmoid function calculated with the current x input data.
    """

    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return  z / (1 + z)

sigmoid = np.vectorize(sigmoid)

def relu(x):
    """
        The relu activation function.

        Parameters
        ---------
        x: float
            Input value to the relu activation function. 
        

        Returns
        --------
        float
            Returns 0 if x is less than 0. Returns x otherwise.
    """
    a = np.maximum(0,x)
    return a

def delta_relu(x):
    """
        The derivative of the relu function.

        Parameters
        ---------
        x: float
            Input value to the relu activation function. 
        

        Returns
        --------
        float
            Returns 1 if x is larger than 0. Returns 0 otherwise. The derivative of the relu function. 
    """
    return np.where(x > 0, 1, 0)

def leaky_relu(x):
    """
        The leaky relu activation function.
        
        Parameters
        ---------
        x: float
            Input value to the leaky relu activation function. 
        

        Returns
        --------
        float
            Returns 0.01 * x if x is less than 0. Returns x otherwise.
    """
    a = np.maximum(0.01*x, x)
    return a 

def delta_leaky_relu(x):
    """
        The derivative of the leaky relu function.

        Parameters
        ---------
        x: float
            Input value to the relu activation function. 
        

        Returns
        --------
        float
            Returns 1 if x is larger than 0. Returns 0.01 otherwise. The derivative of the leaky relu function. 
    """
    return np.where(x > 0, 1, 0.01)