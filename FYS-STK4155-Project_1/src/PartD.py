import numpy as np
from design_matrix import create_design_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sample_data import create_data_samples, DataSamplesType
from linear_model import LinearModel, LinearModelType
from Cross_validation import calculate_stats_with_crossvalidation
from bootstrap import calculate_stats_with_bootstrap
import sys

if __name__ == "__main__": 

    np.random.seed(1234)

    # Deciding between using real or test data
    real_data = True if sys.argv[1] == "real" else False
    if not real_data:
        name_file = "KFOLD_"
        x, y, z = create_data_samples(DataSamplesType.TEST)

    else:
        name_file = "KFOLD_Real_"
        x, y, z = create_data_samples(DataSamplesType.REAL, real_data_n=100)
      

    n_bootstraps = 500
    k = 5
    max_degree = 12

    # Setting up the current linear model to use
    lm = LinearModel(LinearModelType.OLS)

    crossvalidation_error = np.zeros(max_degree)
    bootstrap_error = np.zeros(max_degree)

    # Splitting in test / train for bootstrap algorithm
    x = x.ravel()
    y = y.ravel()
    z = z.ravel().reshape(-1,1)
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2)

    # Scaling data 
    x_train = (x_train - np.mean(x_train))/np.std(x_train)
    x_test = (x_test - np.mean(x_test))/np.std(x_test)
    y_train = (y_train - np.mean(y_train))/np.std(y_train)
    y_test = (y_test - np.mean(y_test))/np.std(y_test)

    # Calculating with both crossvalidation and bootstrap for each degree up to max_degree
    for degree in range(0, max_degree): 
        X = create_design_matrix(x, y, degree)
        crossvalidation_error[degree] = calculate_stats_with_crossvalidation(X, z, k, lm)

        bootstrap_degree, bootstrap_error[degree], bootstrap_bias, bootstrap_variance = calculate_stats_with_bootstrap(x_train, x_test, y_train, y_test, z_train, z_test, n_bootstraps, degree, lm)
    
    # Plotting bootstrap error and crossvalidation error
    plt.figure()
    plt.plot(np.arange(0, degree + 1, 1), crossvalidation_error, label="Crossvalidation error")
    plt.plot(np.arange(0, degree + 1, 1), bootstrap_error, label="Bootstrap error")
    plt.legend()
    plt.xlabel(r"Polynomials")
    plt.ylabel(r"MSE")
    plt.title(r"Comparison of crossvalidation and bootstrap for $k=5$ folds")
    plt.tight_layout()
    plt.savefig(f"figures\{name_file}{str(k)}.pdf")
    plt.show()