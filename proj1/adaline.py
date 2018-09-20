import numpy as np
from neuralnet import NeuralNet

class Adaline(NeuralNet):
    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        self.w_ = rng.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        self.costs_ = []  # for tracking costs

        for _ in range(self.iters):
            output = np.dot(X, self.w_[1:]) + self.w_[0]
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0]  += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.costs_.append(cost)

        return self
