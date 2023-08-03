from FrankeFunction import FrankeFunctionNoised, FrankeFunction
from FFNN import *
from sample_data import create_data_samples, DataSamplesType
from design_matrix import create_design_matrix
from sklearn.model_selection import train_test_split
from mean_square_error import MSE
from activation_functions import * 
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

    # Making plots
    grid_search_hyperparameters_NN(X_train, X_test, y_train, y_test, layers, "Training accuracy (sigmoid)", sigmoid, verbose=True)
    epochs_plot_NN(X_train, y_train, y_train, layers, "Epochs (sigmoid)", 50, 0.01, 0.01, sigmoid)

    # Training neural network with the train data and calculating MSE from both test and train data
    nn = FeedForwardNeuralNetwork(X_train, y_train, layers, 1, 10, epochs=200, eta=0.01, lmbda=0.01)
    nn.train()
    prediction_train = nn.predict_probabilities(X_train)
    prediction_test = nn.predict_probabilities(X_test)
    train_mse = MSE(y_train, prediction_train)
    test_mse = MSE(y_test, prediction_test)
    print(train_mse, test_mse)