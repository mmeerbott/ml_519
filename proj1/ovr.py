import numpy as np
from sgd import SGD

class OVR:
    def __init__(self, classes, eta=0.1, iters=10, random_state=1):
        self.classifiers = { c : SGD(eta,iters,random_state) for c in classes }

    def fit(self, X, y):
        for c, cls in self.classifiers.items():
            z = np.where(c == y, -1, 1)
            cls.fit(X, z)

    def predict(self, X):
        """ return header/key list that were predicted"""
        z = {}
        for c, cls in self.classifiers.items():
            prediction = cls.predict(X)
            z[c] = (prediction, sum(np.ravel(prediction)))

#        results.append(max(z, key=z.get))
            
        return max(z, key=z.get)
