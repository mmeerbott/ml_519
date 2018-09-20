import numpy as np
from neuralnet import NeuralNet

class SGD(NeuralNet):
    def shuffle(self, X, y):
        r = self.rng.permutation(len(y))
        return X[r], y[r]


    def fit(self, X, y):
        self.rng = np.random.RandomState(self.random_state)
        self.w_ = self.rng.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        self.cost_ = []

        for _ in range(self.iters):
            X, y = self.shuffle(X, y)
            cost = []

            for xi, target in zip(X, y):
                cost.append(self.update_weights(xi, target))

            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)

        return self


    def update_weights(self, xi, target):
        output = np.dot(xi, self.w_[1:]) + self.w_[0]
        error  = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0]  += self.eta * error
        cost = error**2 / 2.0
        return cost
