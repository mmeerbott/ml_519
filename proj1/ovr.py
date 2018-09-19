import numpy as np
from .sgd import SGD

class OVR:
    def __init__(self, header, eta=0.1, iters=10, random_state=1):
        self.classifiers = { h : SGD(eta,iters,random_state) for h in header }

    def fit(self, X, y):
        for h, cls in self.classifiers:
            cls.fit(X, y)

    def predict(self, X):
        """ return header/key list that were predicted"""
        results = []
        z = {}
        for h, cls in self.classifiers:
            prediction = cls.predict(X)
            z[h] = (prediction, sum(np.ravel(prediction)))
            results.append(max(z, key=z.get))
            
        return results
