import numpy as np
from neuralnet import NeuralNet

class Perceptron(NeuralNet):
    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        self.w_ = rng.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.costs_ = []  # for tracking

        for _ in range(self.iters):
            errors = 0

            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0]  += update
                errors += int(update != 0.0)  # count # errors for this iteration

            self.costs_.append(errors)

        return self
