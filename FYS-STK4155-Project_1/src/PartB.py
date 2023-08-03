from sample_data import create_data_samples, DataSamplesType
from mean_square_error import MSE
from r2_score import R2score
from design_matrix import create_design_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from linear_model import LinearModel, LinearModelType
from imageio.v2 import imread
import sys

if __name__ == "__main__":

    np.random.seed(1234)

    # Deciding between using real or test data
    real_data = True if sys.argv[1] == "real" else False

    if not real_data:
        name_ols_file = "OLS.pdf"
        name_beta_file = "Beta_values.pdf"
        x, y, z = create_data_samples(DataSamplesType.TEST)

    else:
        name_ols_file = "OLS_Real.pdf"
        name_beta_file = "Beta_values_Real.pdf"
        x, y, z = create_data_samples(DataSamplesType.REAL, real_data_n=100)
        

    # 20% of data is used for test, 80% training
    test_size = 0.2

    # Amount of polynomials to iterate on
    max_polynomial = 5

    mse_values_test = []
    mse_values_train = []
    r2_score_values_test = []
    r2_score_values_train = []
    beta_values = []

    # Setting up the current linear model to use
    lm = LinearModel(LinearModelType.OLS)

    # Doing calculations for each polynomial
    for current_polynominal in range(1, max_polynomial + 1): 
        print(f"at polynomial {current_polynominal} out of {max_polynomial}")
        X = create_design_matrix(x, y, current_polynominal)
        X_train, X_test, y_train, y_test = train_test_split(X, z.ravel(), test_size=test_size)
        
        # Scaling of the data
        X_train = (X_train - np.mean(X_train))/np.std(X_train)
        X_test = (X_test - np.mean(X_test))/np.std(X_test)
        y_train = (y_train - np.mean(y_train))/np.std(y_train)
        y_test = (y_test - np.mean(y_test))/np.std(y_test)

        # Using training data to create beta
        lm.fit(X_train, y_train)

        # Using beta and test data to pretict y
        y_tilde_test = lm.predict(X_test)
        y_tilde_train = lm.predict(X_train)

        # Calculating mean square error and R2 score for each polynomial
        mse_values_test.append(np.mean(MSE(y_test, y_tilde_test)))
        r2_score_values_test.append(np.mean(R2score(y_test, y_tilde_test)))
        mse_values_train.append(np.mean(MSE(y_train, y_tilde_train)))
        r2_score_values_train.append(np.mean(R2score(y_train, y_tilde_train)))

        # Calculating beta
        beta = np.linalg.inv(X_test.T @ X_test) @ X_test.T @ y_tilde_test
        beta_values.append(beta)

    fig, axs = plt.subplots(2)
    fig.tight_layout(pad=5.0)

    # Plotting mean square error for each polynomial
    axs[0].plot(np.arange(1, max_polynomial + 1, 1), mse_values_test, label=r"MSE test")
    axs[0].plot(np.arange(1, max_polynomial + 1, 1), mse_values_train, label=r"MSE train")
    axs[0].legend()
    axs[0].set_title(r"MSE")
    axs[0].set_xlabel(r"Polynomials")
    axs[0].set_ylabel(r"MSE")
    
    # Plotting R2 score for each polynomial
    axs[1].plot(np.arange(1, max_polynomial + 1, 1), r2_score_values_test, label=r"$R^2$ score test")
    axs[1].plot(np.arange(1, max_polynomial + 1, 1), r2_score_values_train, label=r"$R^2$ score train")
    axs[1].legend()
    axs[1].set_title(r"$R^2$ score")
    axs[1].set_xlabel(r"Polynomials")
    axs[1].set_ylabel(r"$R^2$ score")

    plt.savefig(f"figures\{name_ols_file}")
    plt.show()

    #Plotting beta
    for i in range(len(beta_values)):
        length = len(beta_values[i])
        plt.plot(np.linspace(0,length, length), beta_values[i], label=r"Polynomial degree " + f"{i}")

    plt.ylabel(r"$\beta$ values")
    plt.xlabel(r"$\beta$ number")
    plt.legend()
    plt.savefig(f"figures\{name_beta_file}")
    plt.show()