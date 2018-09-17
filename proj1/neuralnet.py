class NeuralNet:
    """ Class for Perceptron, Adaline, and SGD to inherit """

    def __init__(self, eta=0.1, iters=10, random_state=1):
        self.eta = eta
        self.iters = iters
	self.random_state = random_state


    def fit(self, X, y):
        """ Generic fit method """
        rng = np.random.RandomState(self.random_state)
        self.w = rng.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        for _ in range(self.iters):
            errors = 0

            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0]  += update

        return self


    def predict(self, X):
	""" Predict for Perceptron and Adaline """
        z = np.dot(X, self.w_[1:] + self.w_[0]
        return np.where(z>= 0.0, 1, -1)

