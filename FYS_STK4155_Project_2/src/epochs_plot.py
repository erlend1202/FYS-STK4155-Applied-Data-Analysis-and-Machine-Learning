from logistic_regression import *
from FFNN import *
import matplotlib.pyplot as plt

def epochs_plot_NN(X, y, y_exact, layers, plot_title, max_epochs, lmda, eta, func, verbose = False):
    """
        Plotting MSE over epochs of neural network predictions.

        Parameters
        ---------
        X: np.ndarray
            X data used to train the neural network.
        y: np.array
            Y data used to train the neural network.
        y_exact: np.array
            y-data used to calculate MSE, data without noise.
        plot_title: str
            The title text in the plot. Also the filename when saving the plot to disk.
        max_epochs: int
            The maximium number of epochs on the plot.
        lmbda: float
            Lambda hyperparameter to use in the neural network.
        eta: float
            Eta hyperparameter to use in the neural network.
        func: Function
            The activation function the network should use.
        verbose: boolean
            Defaults to false. If true, each value produces for each iteration will be printed to the console.
    """
    mse_values = np.zeros(max_epochs)

    for epoch in range(max_epochs):
        nn = FeedForwardNeuralNetwork(X, y, layers, 1, 10, epochs=epoch, eta=eta, lmbda=lmda, func=func)
        nn.train()
        mse_values[epoch] = MSE(y_exact, nn.predict_probabilities(X))
        if verbose:
            print(f"Testing epoch {epoch}")
    
    plt.figure()
    plt.title(plot_title)
    plt.plot(mse_values)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.savefig(f"figures/{plot_title}")
    print(mse_values)

def epochs_plot_log_reg(X_train, X_test, Y_train, Y_test, plot_title, max_epochs, lmda=0.01, eta=0.1, increment = 5, verbose = False):
    """
        Plotting accuracy over epochs of logistic regression predictions.

        Parameters
        ---------
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
        max_epochs: int
            The maximium number of epochs on the plot.
        lmbda: float
            Lambda hyperparameter to use in the logistic regression algorithm.
        eta: float
            Eta hyperparameter to use in the logistic regression algorithm.
        increment: int
            The increment of the step for the epochs.
        verbose: boolean
            Defaults to false. If true, each value produces for each iteration will be printed to the console.
    """
    num = int(max_epochs/increment)
    acc_values = np.zeros(num)

    for epoch in range(0,max_epochs,increment):
        lg = NumpyLogReg()
        lg.fit(X_train,Y_train, eta=eta, epochs=epoch, M = 5, lmbda=lmda)
        Y_predict = lg.predict(X_test)
        acc = accuracy_score(Y_test, Y_predict, conf=False)
        acc_values[int(epoch/increment)] = acc
        
        if verbose:
            print(f"Testing epoch {epoch}, acc is {acc}")
    
    plt.figure()
    x = np.linspace(0,max_epochs,num)
    plt.title(plot_title)
    plt.plot(x, acc_values)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(f"figures/{plot_title}")