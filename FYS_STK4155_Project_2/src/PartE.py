from sklearn.model_selection import train_test_split
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from logistic_regression import *
from matplotlib.colors import LogNorm
from accuracy_score import accuracy_score
from grid_search import * 
from epochs_plot import *
if __name__ == "__main__":
    # Loading breast cancer dataset
    data = load_breast_cancer()
    X = data.data
    Y = data.target
    
    # Splitting data in train / test
    train_size = 0.8
    test_size = 1 - train_size
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_size, test_size=test_size)

    # Training the logistic regression model with train data
    lg = NumpyLogReg()
    lg.fit(X_train,Y_train, eta=0.1, epochs=100)

    # Prediction results and calculating accuracy score
    Y_predict = lg.predict(X_test)
    acc = accuracy_score(Y_test, Y_predict, conf=True)
    print(acc)
    
    # Creating plots
    grid_search_hyperparameters_log_reg(X_train, X_test, Y_train, Y_test, "Training_accuracy_Logistic", verbose=True)
    epochs_plot_log_reg(X_train, X_test, Y_train, Y_test, "Epochs_Logistic", 200, verbose=True)
    grid_search_hyperparameters_scikit(X_train, X_test, Y_train, Y_test, "Training accuracy logreg (scikit)", verbose=True)