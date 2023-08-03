import numpy as np

# Taken from: https://compphysics.github.io/MachineLearning/doc/pub/week35/html/week35.html 
def create_design_matrix(x, y, max_polynominal):
	"""
		Creates a design matrix based on two dimensional data, with an arbritary number of polynomials as features.
        
		Parameters
        ---------
        x: np.array
            x data feature to be used in the design matrix.
		y: np.array
			y data features to be used in the design matrix.
		max_polynominals: float
			Amount of polynominals of x and y to be entered in the design matrix.

        Returns
		-------
		np.ndarray
			Returns a 2D design matrix with x, y and the selected amount of polynominals. 
	"""
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((max_polynominal+1)*(max_polynominal+2)/2) # Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,max_polynominal+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X


def create_design_matrix_1D(x, max_polynominal):
	"""
		Creates a simple design matrix with one dimensional data.

		Parameters
        ---------
		x: np.array
			Input data to be used in the design matrix.
		max_polynominals: float
			Amount of polynominals of x to be entered in the design matrix.
		
        Returns
		-------
		np.ndarray
			Returns a 1D design matrix with selected amount of polynominals.
	"""
	n = len(x) 
	X = np.ones((n, max_polynominal + 1))      
	for i in range(1, max_polynominal + 1):
		X[:,i] = (x**i).ravel()
	return X