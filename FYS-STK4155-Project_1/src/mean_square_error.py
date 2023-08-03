def MSE(y,y_tilde):
    """
    Calculating the mean square error

    Parameters
    ----------
    y: np.array
        The original y values
    y_tilde: np.array
        The predicted y values

    Returns
    -------
    float
    Returning the calculated mean square error
    """
    sum = 0
    n = len(y)
    for i in range(n):
        sum += (y[i] - y_tilde[i])**2
    return sum/n