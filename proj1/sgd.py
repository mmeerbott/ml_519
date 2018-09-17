import numpy as np

class SGD(NeuralNet):
    def __init__(self, eta=0.1, iters=10, random_state=1, ovr=False):
        super.__init__(eta, iters, 
        self.eta   = eta
        self.iters = iters
        self.random_state = random_state
        self.ovr   = ovr


    def shuffle(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]


    def fit(self, X, y):
        # init weights here TODO 

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
        error = target - output
        self.w_[1:] += self.eta * xi.dot(error)
        slef.w_[0]  += self.eta * error
        cost = 0.5 * (error**2)
        return cost


    def predict(self, X):
        z = np.dot(X, self.w_[1:]) + self.w_[0]
        return np.where(x >= 0.0, 1, -1)
