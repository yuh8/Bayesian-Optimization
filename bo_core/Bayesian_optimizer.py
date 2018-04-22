
import numpy as np
import scipy as sp


class fitGP:
    '''
    Make sure that the columns before the last column
    contain the input data x and the last column
    contains the output y
    '''

    def __init__(self, X):
        self.X = X

    def standardize(self):
        _meanX = np.mean(self.X, axis=0)
        _varX = np.var(self.X, axis=0)
        self.X -= _meanX
        self.X /= np.sqrt(_varX)
        return self.X

    def loglik(self):
        _trainData = self.standardize()
        return _trainData

    def maximize(self):
        pass