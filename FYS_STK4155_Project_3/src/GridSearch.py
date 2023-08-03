from mean_square_error import MSE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from accuracy_score import accuracy_score
from logarithmic_heatmap_plot import create_logarithmic_heatmap_plot
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

def grid_search_hyperparameters_MLP(X_train, X_test, Y_train, Y_test, plot_title, func="relu", alphas = np.logspace(-5,1,7), momentums = np.logspace(-6,0,7), verbose = False):
    """
    Doing a grid search over epochs and the lambdas in ridge regression. Then creating a heatmap with the different MSE results for each lambas and epochs. 
    Parameters
    ----------
    X_train and X_test: np.array
        Input data for training and testing model.
    Y_train and Y_test: np.array
        Correct labels used for training and testing model.
    plot_title: str
        The title text in the plot. Also the filename when saving the plot to disk.
    func: function
        Activation function to use for our model.
    alphas: np.array
        A np.array containing all alpha values for the L2 regularization.
    momentums: np.array
        A np.array containing all the different momentum values.
    verbose: boolean
        Defaults to false. If true, each value produces for each iteration will be printed to the console. 
    """
    acc_values = np.zeros((len(alphas), len(momentums)))
    layers = [18,15,12,8]

    for i, alpha in enumerate(alphas):
        for j, mom in enumerate(momentums):
            model = MLPClassifier(hidden_layer_sizes=layers, activation=func, max_iter=500, alpha=alpha, momentum=mom,)
            model.fit(X_train, Y_train) 
            Y_pred = model.predict(X_test)
            acc = accuracy_score(Y_pred, Y_test, conf=False)
            acc_values[i, j] = acc

            if verbose:
                print(f"alpha:{alpha}, momentum :{mom} gives accuracy {acc}")
    
    create_logarithmic_heatmap_plot(plot_title, "Alpha", "Momentum", acc_values, alphas, momentums, True)


def grid_search_hyperparameters_MLP_F1(X_train, X_test, Y_train, Y_test, plot_title, func="relu", alphas = np.logspace(-5,1,7), momentums = np.logspace(-6,0,7), verbose = False):
    """
    Doing a grid search over epochs and the lambdas in ridge regression. Then creating a heatmap with the different MSE results for each lambas and epochs. 
    Parameters
    ----------
    X_train and X_test: np.array
        Input data for training and testing model.
    Y_train and Y_test: np.array
        Correct labels used for training and testing model.
    plot_title: str
        The title text in the plot. Also the filename when saving the plot to disk.
    func: function
        Activation function to use for our model.
    alphas: np.array
        A np.array containing all alpha values for the L2 regularization.
    momentums: np.array
        A np.array containing all the different momentum values.
    verbose: boolean
        Defaults to false. If true, each value produces for each iteration will be printed to the console. 
    """
    acc_values = np.zeros((len(alphas), len(momentums)))
    layers = [18,15,12,8]

    for i, alpha in enumerate(alphas):
        for j, mom in enumerate(momentums):
            model = MLPClassifier(hidden_layer_sizes=layers, activation=func, max_iter=500, alpha=alpha, momentum=mom,)
            model.fit(X_train, Y_train) 
            Y_pred = model.predict(X_test)
            acc = f1_score(Y_pred, Y_test)
            acc_values[i, j] = acc

            if verbose:
                print(f"alpha:{alpha}, momentum :{mom} gives f1 score {acc}")
    
    create_logarithmic_heatmap_plot(plot_title, "Alpha", "Momentum", acc_values, alphas, momentums, True)