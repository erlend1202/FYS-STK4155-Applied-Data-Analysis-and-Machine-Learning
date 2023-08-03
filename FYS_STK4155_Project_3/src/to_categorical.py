import numpy as np

def to_categorical(integer_vector):
    """
        One-hot in numpy
    """
    n_inputs = len(integer_vector)
    n_categories = int(np.max(integer_vector) + 1)
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1
    
    return onehot_vector