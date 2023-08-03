from FFNN import FeedForwardNeuralNetwork
from mean_square_error import MSE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from activation_functions import *
from sklearn.model_selection import train_test_split
from to_categorical import *
from logistic_regression import *
from SGD import *
from accuracy_score import accuracy_score
from logarithmic_heatmap_plot import create_logarithmic_heatmap_plot
from sklearn.linear_model import LogisticRegression

def grid_search_hyperparameters_ridge(x, y, z, plot_title, func, lambdas = np.logspace(-5,1,7), epochs = np.linspace(10,190,7, dtype='int'), verbose = False):
    """
    Doing a grid search over epochs and the lambdas in ridge regression. Then creating a heatmap with the different MSE results for each lambas and epochs. 

    Parameters
    ----------
    x: np.array
        x-data used in the ridge regression algorithm.
    y: np.array
        y-data used in the ridge regression algorithm with noise.
    y_exact: np.array
        y-data used in the ridge regression algorithm without noise.
    plot_title: str
        The title text in the plot. Also the filename when saving the plot to disk.
    func: function
        Expected SGD_Ridge function, although any function can be used.
    lambdas: np.array
        A np.array containing all lambda values for the ridge regression.
    epochs: np.array
        A np.array containing all the different epochs for the ridge regression.
    verbose: boolean
        Defaults to false. If true, each value produces for each iteration will be printed to the console. 
    """
    mse_values = np.zeros((len(lambdas), len(epochs)))

    for i, epoch in enumerate(epochs):
        for j, lmd in enumerate(lambdas):
            if func == SGD_Ridge:
                xnew,y_tilde = func(x,y,z, Niterations=epoch, momentum=0.01, M=5, eta=0.1, lmbda=lmd, plot=False)
            else:
                print("wrong function, will only work for SGD_Ridge")
                return 0
            mse = MSE(z, y_tilde)
            mse_values[i, j] = mse

            if verbose:
                print(f"epoch:{epoch}, Lambda :{lmd} gives mse {mse}")
    
    create_logarithmic_heatmap_plot(plot_title, "$\lambda$", "$Epochs$", mse_values, epochs, lambdas, True)

def grid_search_hyperparameters_SGD_epochs(x, y, z, plot_title, func, batch_size = np.linspace(1, 19, 10, dtype='int'), epochs = np.linspace(10, 190, 10, dtype='int'), verbose = False):
    """
    Doing a grid seach over batch size and epochs in with the SGD algorithm. Then creating a heatmap with the different MSE results for each batch size and epochs.

    Parameters
    ----------
    x: np.array
        x-data used in the SGD algorithm.
    y: np.array
        y-data used in the SGD algorithm with noise.
    y_exact: np.array
        y-data used in the SGD algorithm without noise.
    plot_title: str
        The title text in the plot. Also the filename when saving the plot to disk.
    func: function
        Expected SGD_Tuned olr SGD function, although any function can be used.
    batch_size: np.array
        A np.array containing all the different batch sizes that should be used.
    epochs: np.array
        A np.array containing all the different epochs for that should be used.
    verbose: boolean
        Defaults to false. If true, each value produces for each iteration will be printed to the console. 
    """

    mse_values = np.zeros((len(batch_size), len(epochs)))

    for i, epoch in enumerate(epochs):
        for j, M in enumerate(batch_size):
            if func == SGD_Tuned or func == SGD:
                xnew, y_tilde = func(x,y,z, Niterations=epoch, momentum=0.01, M=M, eta=0.1, plot=False)
            else:
                print("wrong function, will only work for SGD")
                return 0
            mse = MSE(z, y_tilde)
            mse_values[i, j] = mse

            if verbose:
                print(f"epoch:{epoch}, Batch size:{M} gives mse {mse}")
    
    create_logarithmic_heatmap_plot(plot_title, "$Batch size$", "$Epochs$", mse_values, batch_size, epochs, True)

def grid_search_hyperparameters_SGD(x, y, z, plot_title, func, verbose = False, value_text_in_cells = True):
    """
    Doing a grid seach over the learning rate and the momentum in with the SGD algorithm. Then creating a heatmap with the different MSE results for each learning rate and momentum.

    Parameters
    ----------
    x: np.array
        x-data used in the SGD algorithm.
    y: np.array
        y-data used in the SGD algorithm with noise.
    y_exact: np.array
        y-data used in the SGD algorithm without noise.
    plot_title: str
        The title text in the plot. Also the filename when saving the plot to disk.
    func: function
        Expected SGD_Tuned, GD or SGD function, although any function can be used.
    verbose: boolean
        Defaults to false. If true, each value produces for each iteration will be printed to the console. 
    """

    """    
    if func == GD:
        learning_rates = np.logspace(-5,-1,5)
        momentums = np.logspace(-5,-1,5)
    elif func == SGD:
        momentums = np.logspace(-5,1,7)
        learning_rates = np.logspace(-5,1,3)
    else:
        learning_rates = np.logspace(-5,1,7)
        momentums = np.logspace(-5,1,7)"""

    learning_rates = np.logspace(-5,1,7)
    momentums = np.logspace(-5,1,7)
    mse_values = np.zeros((len(momentums), len(learning_rates)))

    for i, mom in enumerate(momentums):
        for j, eta in enumerate(learning_rates):
            if func == SGD_Tuned or func == SGD:
                xnew, y_tilde = func(x, y, z, Niterations=20, momentum=mom, M=5, eta=eta, plot=False)
            else:
                xnew,y_tilde = func(x, y, z, Niterations=20, momentum=mom, eta=eta, plot=False)
            mse = MSE(z, y_tilde)
            if mse > 5000:
                mse_values[i, j] = None 
            else:
                mse_values[i,j] = mse

            if verbose:
                print(f"eta:{eta}, momentum:{mom} gives mse {mse}")

    create_logarithmic_heatmap_plot(plot_title, "$Momentum$", "$\eta$", mse_values, learning_rates, momentums, value_text_in_cells=value_text_in_cells)
    plt.gcf().autofmt_xdate()

def grid_search_hyperparameters_log_reg(X_train, X_test, Y_train, Y_test, plot_title, M = 10, epochs = 100, eta_vals = np.logspace(-5, 1, 7), lmd_vals = np.logspace(-5, 1, 7), verbose = False):
    """
    Doing a grid seach over the eta values and the lambda values in with the logistic regression. Then creating a heatmap with the different accuracy scores for each eta and lambda.

    Parameters
    ----------
    X_train: np.array
        X training values from a dataset splitted in train/test values.
    X_test: np.array
        X test values from a dataset splitted in train/test values.
    Y_train: np.array
        Y training values from a dataset splitted in train/test values.
    Y_test: np.array
        Y testing values from a dataset splitted in train/test values.
    plot_title: str
        The title text in the plot. Also the filename when saving the plot to disk.
    M: int
        The M value in logistic regression. Defaults to 10.
    epochs: int
        The number of epochs that should be used to train the logistic regression model. Defaults to 100
    eta_vals: np.array
        A np.array containing all eta values.
    lmd_vals: np.array
        A np.array containing all lambda values.
    verbose: boolean
        Defaults to false. If true, each value produces for each iteration will be printed to the console. 
    """
    acc_values = np.zeros((len(eta_vals), len(lmd_vals)))

    for i, eta in enumerate(eta_vals):
        for j, lmd in enumerate(lmd_vals):
            lg = NumpyLogReg()
            lg.fit(X_train,Y_train, eta=eta, epochs=epochs, M = M, lmbda=lmd)
            Y_predict = lg.predict(X_test)
            acc = accuracy_score(Y_test, Y_predict, conf=False)

            acc_values[i, j] = acc

            if verbose:
                print(f"eta:{eta}, lambda:{lmd} gives acc {acc}")

    create_logarithmic_heatmap_plot(plot_title, "$\lambda$", "$\eta$", acc_values, eta_vals, lmd_vals, True)


def grid_search_hyperparameters_scikit(X_train, X_test, Y_train, Y_test, plot_title, epochs = np.linspace(10,130,7, dtype='int'), lmd_vals = np.logspace(-5, 1, 7), verbose = False):
    """
    Doing a grid seach over the eta values and the lambda values in with the logistic regression. Then creating a heatmap with the different accuracy scores for each eta and lambda.

    Parameters
    ----------
    X_train: np.array
        X training values from a dataset splitted in train/test values.
    X_test: np.array
        X test values from a dataset splitted in train/test values.
    Y_train: np.array
        Y training values from a dataset splitted in train/test values.
    Y_test: np.array
        Y testing values from a dataset splitted in train/test values.
    plot_title: str
        The title text in the plot. Also the filename when saving the plot to disk.
    M: int
        The M value in logistic regression. Defaults to 10.
    epochs: int
        The number of epochs that should be used to train the logistic regression model. Defaults to 100
    eta_vals: np.array
        A np.array containing all eta values.
    lmd_vals: np.array
        A np.array containing all lambda values.
    verbose: boolean
        Defaults to false. If true, each value produces for each iteration will be printed to the console. 
    """
    acc_values = np.zeros((len(epochs), len(lmd_vals)))

    for i, epoch in enumerate(epochs):
        for j, lmd in enumerate(lmd_vals):
            clf = LogisticRegression(C=lmd, max_iter=epoch).fit(X_train, Y_train)
            Y_predict = clf.predict(X_test)
            acc = accuracy_score(Y_test, Y_predict, conf=False)

            acc_values[i, j] = acc

            if verbose:
                print(f"epoch:{epoch}, lambda:{lmd} gives acc {acc}")

    create_logarithmic_heatmap_plot(plot_title, "$\lambda$", "iterations", acc_values, epochs, lmd_vals, True)



def grid_search_hyperparameters_NN_classification(X_train, X_test, Y_train, Y_test, plot_title, layers=[10], func=sigmoid, eta_vals = np.logspace(-5, 1, 7), lmd_vals = np.logspace(-5, 1, 7), verbose = False):
    """
    Doing a grid seach over the eta values and the lambda values in a neural network used for classification. Then creating a heatmap with the different accuracy scores for each eta and lambda.

    Parameters
    ----------
    X_train: np.array
        X training values from a dataset splitted in train/test values.
    X_test: np.array
        X test values from a dataset splitted in train/test values.
    Y_train: np.array
        Y training values from a dataset splitted in train/test values.
    Y_test: np.array
        Y testing values from a dataset splitted in train/test values.
    plot_title: str
        The title text in the plot. Also the filename when saving the plot to disk.
    layers: array
        An array representing the different hidden layers in the neural network.
    func: function
        The activation function to use in the neural network. Expecting either the sigmoid / relu or leaky_relu function, although any function can be used.
    eta_vals: np.array
        A np.array containing all eta values.
    lmd_vals: np.array
        A np.array containing all lambda values.
    verbose: boolean
        Defaults to false. If true, each value produces for each iteration will be printed to the console. 
    """
    acc_values = np.zeros((len(eta_vals), len(lmd_vals)))
    #Y_train = to_categorical(Y_train.T).T
    for i, eta in enumerate(eta_vals):
        for j, lmd in enumerate(lmd_vals):
            nn = FeedForwardNeuralNetwork(X_train, Y_train, layers, 2, batch_size=5, epochs=10, eta=eta, lmbda=lmd, func=func, problem="classification")
            nn.train()
            Y_predict = nn.predict(X_test)
            acc = accuracy_score(Y_test, Y_predict)

            acc_values[i, j] = acc

            if verbose:
                print(f"eta:{eta}, lambda:{lmd} gives acc {acc}")

    create_logarithmic_heatmap_plot(plot_title, "$\lambda$", "$\eta$", acc_values, eta_vals, lmd_vals, True)

def grid_search_hyperparameters_NN(X_train, X_test, y_train, y_test, layers, plot_title, func, n_categories = 1, batch_size = 5, epochs = 5, eta_vals = np.logspace(-5, 1, 7), lmd_vals = np.logspace(-5, 1, 7), verbose = False):

    """
    Doing a grid seach over the eta values and the lambda values in a neural network. Then creating a heatmap with the different accuracy scores for each eta and lambda.

    Parameters
    ----------
    X_train: np.array
        X training values from a dataset splitted in train/test values.
    X_test: np.array
        X test values from a dataset splitted in train/test values.
    Y_train: np.array
        Y training values from a dataset splitted in train/test values.
    Y_test: np.array
        Y testing values from a dataset splitted in train/test values.
    plot_title: str
        The title text in the plot. Also the filename when saving the plot to disk.
    layers: array
        An array representing the different hidden layers in the neural network.
    func: function
        The activation function to use in the neural network. Expecting either the sigmoid / relu or leaky_relu function, although any function can be used.
    n_categories: int
        The number of output layers in the neural network.
    batch_size: int
        The batch size used in the neural network.
    epochs: int
        The number of epochs in the training process of the neural network.
    eta_vals: np.array
        A np.array containing all eta values.
    lmd_vals: np.array
        A np.array containing all lambda values.
    verbose: boolean
        Defaults to false. If true, each value produces for each iteration will be printed to the console. 
    """
    mse_values = np.zeros((len(eta_vals), len(lmd_vals)))

    for i, eta in enumerate(eta_vals):
        for j, lmd in enumerate(lmd_vals):
            nn = FeedForwardNeuralNetwork(X_train, y_train, layers, n_categories, batch_size, epochs = epochs, eta = eta, lmbda = lmd, func = func)
            nn.train()
            y_tilde = nn.predict_probabilities(X_test)
            mse = MSE(y_test, y_tilde)
            mse_values[i, j] = mse

            if verbose:
                print(f"eta:{eta}, lambda:{lmd} gives mse {mse}")

    create_logarithmic_heatmap_plot(plot_title, "$\lambda$", "$\eta$", mse_values, eta_vals, lmd_vals, True)

def grid_search_layers(X_train, X_test, Y_train, Y_test, plot_title, func=sigmoid, verbose = False):
    """
    Doing a grid seach over the number of layers and the layer size in a neural network. Then creating a heatmap with the different accuracy scores for each layer and layer size.

    Parameters
    ----------
    X_train: np.array
        X training values from a dataset splitted in train/test values.
    X_test: np.array
        X test values from a dataset splitted in train/test values.
    Y_train: np.array
        Y training values from a dataset splitted in train/test values.
    Y_test: np.array
        Y testing values from a dataset splitted in train/test values.
    plot_title: str
        The title text in the plot. Also the filename when saving the plot to disk.
    layers: array
        An array representing the different hidden layers in the neural network.
    func: function
        The activation function to use in the neural network. Expecting either the sigmoid / relu or leaky_relu function, although any function can be used.
    verbose: boolean
        Defaults to false. If true, each value produces for each iteration will be printed to the console. 
    """
    num_layers = np.linspace(1,10,10, dtype='int')
    size_layers = np.linspace(1,10,10, dtype='int')

    acc_values = np.zeros((len(num_layers), len(size_layers)))
    Y_train = to_categorical(Y_train)
    for i, num in enumerate(num_layers):
        for j, size in enumerate(size_layers):
            layers = [size]*num
            nn = FeedForwardNeuralNetwork(X_train, Y_train, layers=layers, n_categories=2, batch_size=5, epochs=10, eta=1, lmbda=0.0001, func=func, problem="classification")
            nn.train()
            Y_predict = nn.predict(X_test)
            acc = accuracy_score(Y_test, Y_predict)

            acc_values[i, j] = acc

            if verbose:
                print(f"num_layers:{num}, size:{size} gives acc {acc}")

    create_logarithmic_heatmap_plot(plot_title, "Number of layers", "Size of layers", acc_values, num_layers, size_layers, True)