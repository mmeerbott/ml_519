# Michael Meerbott

import numpy as np
# TODO may need more imports

class Perceptron(NeuralNet):
    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        self.w = rng.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        for _ in range(self.iters):
            errors = 0  # XXX what's this for?

            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0]  += update

        return self

    
    def predict(self, X):
        z = np.dot(X, self.w_[1:] + self.w_[0]
        return np.where(z>= 0.0, 1, -1)
