from mean_square_error import MSE
import numpy as np

def R2score(y, y_tilde):
    """
    Calculating the r2 score

    Parameters
    ----------
    y: np.array
        The original y values
    y_tilde: np.array
        The predicted y values

    Returns
    -------
    float
    Returning the calculated r2 score
    """
    y_mean = np.mean(y)
    mse = MSE(y,y_tilde)
    sum = 0
    n = len(y)
    for i in range(n):
            sum += (y[i] - y_mean)**2
    sum /= n 
    return 1 - mse/sum