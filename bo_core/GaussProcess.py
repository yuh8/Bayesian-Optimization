
import numpy as np
from scipy.optimize import minimize as spmin
from .Functions import *


class GP:
    '''
    Fitting and prediction with Gaussian process with sqaure exponential kernel
    '''

    def __init__(self, X, y):
        # Input data rows are features, columns are data samples
        # d by N array
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(X.shape) < 2:
            raise ValueError('the number of data points must be at least 2')
        # Training data sample size
        self.N = np.size(X, 0)
        # Demean train data
        self.X, self.meanX = demean(X)

        # Output data is a 1D numpy array
        y = np.asarray(y, dtype=float)
        self.y, self.meany = demean(y)
        # Check training data consistency
        n2 = len(self.y)
        if self.N != n2:
            raise ValueError('the number of output and input training data samples must be equal')

    # Compute marginal log-likelihood
    def negloglik(self, par):
        _, Ks, invKs = choleInvKs(par, self.X, covSE)
        # Compute negative log-likelihood Eq 2.30 of RW book
        negloglik = self.N / 2 * np.log(2 * np.pi)
        negloglik += 1 / 2 * np.dot(np.dot(self.y, invKs), self.y)
        negloglik += 1 / 2 * np.log(np.linalg.det(Ks))
        return negloglik

    # method for computing the derivative of the negloglike
    def der_negloglik(self, par):
        K, Ks, invKs = choleInvKs(par, self.X, covSE)
        der = np.zeros(len(par))
        # Eq.5.9 of RW book
        alpha = np.dot(invKs, self.y)
        alpha2 = np.outer(alpha, alpha)
        temp = alpha2 - invKs
        # Derivative w.r.t par[0]
        der[0] = covSE(par, self.X, self.X, trainmode=1)
        der[0] = 1 / 2 * np.trace(np.dot(temp, der[0]))
        # Derivative w.r.t par[1]
        der[1] = covSE(par, self.X, self.X, trainmode=2)
        der[1] = 1 / 2 * np.trace(np.dot(temp, der[1]))
        # Derivative w.r.t par[2]
        der[2] = 2 * par[2] * np.eye(self.N)
        der[2] = 1 / 2 * np.trace(np.dot(temp, der[2]))
        return der

    @property
    def minimize(self):
        par0 = np.array([0.01, 0.01, 0.01])
        par_bar = spmin(self.negloglik, par0, method='BFGS', jac=self.der_negloglik, options={'xtol': 1e-6, 'disp': True})
        return par_bar, self.meanX

    # Posterior prediction
    def GP_predict(self, par_bar, Xpre):
        # Testing data sample size
        if len(Xpre.shape) == 1:
            Xpre = Xpre.reshape(1, -1)
        Npre = np.size(Xpre, 0)
        # Demean test data
        temp = np.tile(self.meanX, (Npre, 1))
        Xpre -= temp
        mean_Ypre, var_Ypre = Predict(par_bar, self.X, self.y, self.meany, Xpre, covSE)
        return mean_Ypre, var_Ypre
