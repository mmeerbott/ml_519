import numpy as np

class NeuralNet:
    """ Class for Perceptron, Adaline, and SGD to inherit """

    def __init__(self, eta=0.1, iters=10, random_state=1):
        self.eta = eta
        self.iters = iters
        self.random_state = random_state


    def fit(self, X, y):
        pass


    def predict(self, X):
        """ Predict for Perceptron, Adaline, SGD """
        z = np.dot(X, self.w_[1:]) + self.w_[0]
        return np.where(z >= 0.0, 1, -1)
