import numpy as np
from neuralnet import NeuralNet

class SGD(NeuralNet):
    def __init__(self, eta=0.01, n_iter=10, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.w_initialized = False


    def shuffle(self, X, y):
        r = self.rng.permutation(len(y))
        return X[r], y[r]


    def init_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True


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
        try:
            cost = error**2 / 2.0
        except FloatingPointError:
            sys.exit("Diverging. Please use a smaller eta.")
        return cost
