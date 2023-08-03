import numpy as np
from random import random

def FrankeFunction(x,y):
    """
    Calculates the franke function
    Parameters
    ----------
    x: np.array
        A meshgrid array with x sample points
    y: np.array
        A meshgrid array with y sample points
    Returns
    -------
    np.array
        Array with franke function points calculated from x and y inputs
    """

    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    return term1 + term2 + term3 + term4

def FrankeFunctionNoised(x, y, max_noise): 
    """
    Calculates the franke function with an added noise
    Parameters
    ----------
    x: np.array
        A meshgrid array with x sample points
    y: np.array
        A meshgrid array with y sample points
    max_noise: float
        Maximum amount of noise added to the values from the franke function.
        Noise is normal distributed N~[0, max_noise]
    Returns
    -------
    np.array
        Array with franke function points calculated from x and y inputs with 
        a added noise
    """

    ff = FrankeFunction(x, y)
    noise = np.random.normal(0, max_noise, len(x)*len(x))
    noise = noise.reshape(len(x), len(x))

    return ff + noise