import numpy as np
import matplotlib.pyplot as plt
from FrankeFunction import FrankeFunctionNoised
from design_matrix import create_design_matrix
from sklearn.model_selection import train_test_split
from mean_square_error import MSE
from sample_data import create_data_samples, DataSamplesType

def plot_MSE_variance(x,y, name):

    # Number of polynomials
    num = 14

    # Noise value for franke function
    val = 0.1
    mse_arr_train = np.zeros(num-1)
    mse_arr_test = np.zeros(num-1)
    for i in range(1,num):
        print(f"polynomial {i} out of {num}")
        z = FrankeFunctionNoised(x,y, val)
        z = z.ravel()
        X = create_design_matrix(x,y, i)
        X_train, X_test, z_train, z_test= train_test_split(X, z, test_size=0.2)
        
        beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ z_train
        z_tilde_test = X_test @ beta
        z_tilde_train = X_train @ beta

        mse_arr_train[i-1] = MSE(z_train, z_tilde_train)
        mse_arr_test[i-1] = MSE(z_test, z_tilde_test)

    n_arr = np.array([i for i in range(1,num)])
    plt.plot(n_arr, mse_arr_train, label= "MSE_train")
    plt.plot(n_arr, mse_arr_test, label= "MSE_test")
    plt.xlabel(r"Polynomials")
    plt.ylabel(r"MSE")
    plt.legend()
    plt.savefig(f"figures\{name}")
    plt.show()

if __name__ == "__main__":
    np.random.seed(1234)

    # Choosing between test and real data
    real_data = True
    if not real_data:
        name_file = "MSE_test_train.pdf"
        x = np.arange(0, 1, 0.01)  
        y = np.arange(0, 1, 0.01)
        x, y = np.meshgrid(x,y)
    else:
        name_file = "MSE_test_train_Real.pdf"
        x, y, z = create_data_samples(DataSamplesType.REAL)

    plot_MSE_variance(x,y, name_file)