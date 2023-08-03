import numpy as np
from sklearn.metrics import confusion_matrix

def accuracy_score(Y_test, Y_pred, conf=True):
    """
        Calculates the accuracy score for a classification problems.

        Parameters
        ---------
        Y_test: np.array
            Test data used to make the predictions.
        
        Y_pred: np.array
            The prediction calculated from the model.

        conf: boolean
            Prints the confution matrix in addition to returning the accuracy. Defaults to True.

        Returns
        -------
        float
            Accuracy score between 0 - 1 where 1 is the best score.
    """
    if conf: print(confusion_matrix(Y_test, Y_pred))
    return np.sum(Y_test == Y_pred) / len(Y_test)