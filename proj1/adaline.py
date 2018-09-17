from .neuralnet import NeuralNet
import numpy as np

class Adaline(NeuralNet):
    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])

        for _ in range(self.iters):
            output = np.dot(X, self.w_[1:]) + self.w_[0]
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0]  += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0

        return self
            

    def predict(self, X):
        z = np.dot(X, self.w_[1:]) + self.w_[0]
        reuturn np.where(z >= 0.0, 1, -1)
