from FFNN import *
import numpy as np
from design_matrix import *
from sklearn.model_selection import train_test_split
from sample_data import *
from grid_search import grid_search_hyperparameters_NN
from epochs_plot import *

if __name__ == "__main__":
    # Running the neural network on train and test dataset created from franke function
    x, y, z = create_data_samples(DataSamplesType.TEST)
    X = create_design_matrix(x, y, 5)

    # Splitting data in train / test
    X_train, X_test, y_train, y_test = train_test_split(X, z.ravel(), test_size=0.2)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    # Representing the hidden layers in the neural network
    layers = [10]

    # Setting a seed for getting the same result consecutively
    np.random.seed(40)
    nn = FeedForwardNeuralNetwork(X_train, y_train, layers, eta=0.1, lmbda = 0.1, epochs=5, func=leaky_relu)
    nn.train()
    y_pred = nn.predict_probabilities(X_test)
    print(MSE(y_test, y_pred))
    
    # Creating plots
    grid_search_hyperparameters_NN(X_train, X_test, y_train, y_test, layers, "Training accuracy (RELU)", relu, verbose=True)
    grid_search_hyperparameters_NN(X_train, X_test, y_train, y_test, layers, "Training accuracy (Leaky RELU)", leaky_relu, verbose=True)
    epochs_plot_NN(X_train, y_train, y_train, layers, "Epochs (RELU)", 50, 0.01, 0.01, relu)
    #epochs_plot_NN(X, y_train, y_train, layers, "Epochs (Leaky RELU)", 50, 0.01, 0.01, leaky_relu)
    epochs_plot_NN(X_train, y_train, y_train, layers, "Epochs (Leaky RELU)", 50, 1e-05, 10, leaky_relu)
