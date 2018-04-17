import numpy as np
import scipy as sp


class fitGP:
    '''
    Make sure that the columns before the last column
    contain the input data x and the last column
    contains the output y
    '''

    def __init__(self, X):
        '''
        for now the only input parameter is X
        '''
        self.X = X

    def standardize(self):
        _meanX = np.mean(X,)
