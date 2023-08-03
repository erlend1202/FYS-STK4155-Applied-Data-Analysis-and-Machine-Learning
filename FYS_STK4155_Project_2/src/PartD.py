import numpy as np
from FFNN import *
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from to_categorical import to_categorical
from accuracy_score import accuracy_score
from grid_search import *

#For eta and lambda
#Best seems to be eta=1, lmbda = 0.0001
def grid_search_hyperparameters(plot_title, layers=[10], func=sigmoid, eta_vals = np.logspace(-5, 1, 7), lmd_vals = np.logspace(-5, 1, 7), verbose = False):
    data = load_breast_cancer()
    X = data.data
    Y = data.target
    train_size = 0.8
    test_size = 1 - train_size
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_size,
                                                    test_size=test_size)
    acc_values = np.zeros((len(eta_vals), len(lmd_vals)))
    
    Y_train = to_categorical(Y_train)
    for i, eta in enumerate(eta_vals):
        for j, lmd in enumerate(lmd_vals):
            nn = FeedForwardNeuralNetwork(X_train, Y_train, layers, 2, batch_size=5, epochs=10, eta=eta, lmbda=lmd, func=func, problem="classification")
            nn.train()
            Y_predict = nn.predict(X_test)
            acc = accuracy_score(Y_test, Y_predict)

            acc_values[i, j] = acc

            if verbose:
                print(f"eta:{eta}, lambda:{lmd} gives acc {acc}")

    def array_elements_to_string(arr):
        new_arr = []

        for element in arr:
            new_arr.append(str(element))
        
        return new_arr
    
    def show_values_in_heatmap(heatmap, axes, text_color = "white"):
        for i in range(len(heatmap)):
            for j in range(len(heatmap[0])):
                axes.text(j, i, np.round(heatmap[i, j], 2), ha="center", va="center", color=text_color)

    labels_x = array_elements_to_string(eta_vals)
    labels_y = array_elements_to_string(lmd_vals)

    plt.figure()
    show_values_in_heatmap(acc_values, plt.gca())
    plt.title(plot_title)
    plt.xticks(np.arange(0, len(eta_vals)), labels_x)
    plt.yticks(np.arange(0, len(lmd_vals)), labels_y)
    plt.xlabel("$\lambda$")
    plt.ylabel("$\eta$")
    plt.imshow(acc_values, norm=LogNorm())
    plt.colorbar()
    plt.savefig(f"figures/{plot_title}")



def grid_search_layers(plot_title, func=sigmoid, verbose = False, sc=False):
    data = load_breast_cancer()
    X = data.data
    Y = data.target
    train_size = 0.8
    test_size = 1 - train_size
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_size,
                                                    test_size=test_size)
    
    num_layers = np.linspace(1,10,10, dtype='int')
    size_layers = np.linspace(1,10,10, dtype='int')

    acc_values = np.zeros((len(num_layers), len(size_layers)))
    
    if not sc:
        Y_train = to_categorical(Y_train)

    for i, num in enumerate(num_layers):
        for j, size in enumerate(size_layers):
            layers = [size]*num
            if sc:            
                mlp = MLPClassifier(hidden_layer_sizes=layers, activation='relu', solver='adam', max_iter=500)
                mlp.fit(X_train,Y_train)
                Y_predict = mlp.predict(X_test)
            else:
                nn = FeedForwardNeuralNetwork(X_train, Y_train, layers=layers, n_categories=2, batch_size=5, epochs=10, eta=1, lmbda=0.0001, func=func, problem="classification")
                nn.train()
                Y_predict = nn.predict(X_test)
            
            acc = accuracy_score(Y_test, Y_predict)
            acc_values[i, j] = acc

            if verbose:
                print(f"num_layers:{num}, size:{size} gives acc {acc}")

    def array_elements_to_string(arr):
        new_arr = []

        for element in arr:
            new_arr.append(str(element))
        
        return new_arr
    
    def show_values_in_heatmap(heatmap, axes, text_color = "white"):
        for i in range(len(heatmap)):
            for j in range(len(heatmap[0])):
                axes.text(j, i, np.round(heatmap[i, j], 2), ha="center", va="center", color=text_color)

    labels_x = array_elements_to_string(num_layers)
    labels_y = array_elements_to_string(size_layers)

    plt.figure()
    show_values_in_heatmap(acc_values, plt.gca())
    plt.title(plot_title)
    plt.xticks(np.arange(0, len(num_layers)), labels_x)
    plt.yticks(np.arange(0, len(size_layers)), labels_y)
    plt.xlabel("number of layers")
    plt.ylabel("size of layers")
    plt.imshow(acc_values, norm=LogNorm())
    plt.colorbar()
    plt.savefig(f"figures/{plot_title}")



if __name__ == "__main__": 
    # Representing the hidden layers in the neural network
    layers = [10]

    # Loading breast cancer dataset
    data = load_breast_cancer()
    X = data.data
    Y = data.target


    # Splitting data in train / test
    train_size = 0.8
    test_size = 1 - train_size
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_size, test_size=test_size)
    Y_train = to_categorical(Y_train)
    
    mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
    mlp.fit(X_train,Y_train)
    Y_predict = mlp.predict(X_test)

    #Important to change n_categories to 2 and problem to anything else than regression
    nn = FeedForwardNeuralNetwork(X_train, Y_train, layers, 2, 5, epochs=100, eta=1, lmbda=0.001, func=sigmoid, problem="classification")
    nn.train()
    Y_predict = nn.predict(X_test)
    print(accuracy_score(Y_test, Y_predict))

    grid_search_hyperparameters("Prediction accuracy (sigmoid)", func=sigmoid, verbose=True)
    grid_search_layers("Prediction accuracy different layers (sigmoid)", func=sigmoid, verbose=True)

    grid_search_hyperparameters("Prediction accuracy (relu)", func=relu, verbose=True)
    grid_search_layers("Prediction accuracy different layers (relu)", func=relu, verbose=True)

    grid_search_hyperparameters("Prediction accuracy (leaky_relu)", func=leaky_relu, verbose=True)
    grid_search_layers("Prediction accuracy different layers (leaky_relu)", func=leaky_relu, verbose=True)

    grid_search_layers("Prediction accuracy different layers (scikit)", func=leaky_relu, verbose=True, sc=True)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_size, test_size=test_size)
    Y_train = to_categorical(Y_train)

    # Creating plots
    grid_search_hyperparameters_NN_classification(X_train, X_test, Y_train, Y_test, "Prediction accuracy (sigmoid) classification", func = sigmoid, verbose = True)
    grid_search_layers(X_train, X_test, Y_train, Y_test, "Prediction accuracy different layers (sigmoid) classification", func = sigmoid, verbose = True)
