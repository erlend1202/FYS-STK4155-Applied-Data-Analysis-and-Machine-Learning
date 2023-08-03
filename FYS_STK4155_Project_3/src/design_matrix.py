import numpy as np

# Taken from: https://compphysics.github.io/MachineLearning/doc/pub/week35/html/week35.html 
def create_design_matrix(x, y, max_polynominal):
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