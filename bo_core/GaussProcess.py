
import numpy as np
from scipy.optimize import minimize as spmin
from .Functions import *


class GP:
    '''
    Fitting and prediction with Gaussian process with squared exponential kernel
    '''

    def __init__(self, X, y):
        # Input data rows are samples, columns are features
        # N by D array
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(X.shape) < 2 or len(y) < 2:
            raise ValueError('the number of data points must be at least 2')
        # Training data sample size
        self.N = np.size(X, 0)
        # Standardize train data
        self.X, self.meanX, self.stdX = demean(X)

        # Output data is a 1D numpy array
        y = np.asarray(y, dtype=float)
        self.y, self.meany = demean(y, std=0)
        # Check training data consistency
        n2 = len(self.y)
        if self.N != n2:
            raise ValueError('the number of output and input training data samples must be equal')

    # Compute the negative marginal log-likelihood
    def negloglik(self, par):
        par = np.squeeze(par)
        _, Ks, invKs = choleInvKs(par, self.X, covSE)
        # Compute negative log-likelihood Eq 2.30 of RW book
        negloglik = self.N / 2 * np.log(2 * np.pi)
        negloglik += 1 / 2 * np.dot(np.dot(self.y.T, invKs), self.y)
        negloglik += 1 / 2 * logdetX(Ks)
        return negloglik

    # Fit gaussian process
    def fit(self, nstarts=10):
        '''
        nstarts = number of random starts
        '''
        # Be careful of the lazy coding of number of hyper parameters
        d = 3
        min_fun = np.inf
        for i in range(0, nstarts):
            par0 = np.random.rand(d)
            # Be careful the output of scipy minimize is an ndarray
            res = spmin(self.negloglik, par0, method='L-BFGS-B', options={'disp': False})
            # Reject non-successful runs
            if not res.success:
                continue
            # Choose the start yielding the Max LL
            if res.fun < min_fun:
                min_fun = res.fun
                par = res.x
        _, _, self.invKs = choleInvKs(par, self.X, covSE)
        return par, self.meanX, self.stdX

    # Posterior prediction
    def predict(self, par, Xpre):
        # Testing data sample size
        Xpre = np.asarray(Xpre)
        if len(Xpre.shape) == 1:
            Xpre = Xpre.reshape(1, -1)
        Npre = Xpre.shape[0]
        # standardize test data
        Xpre -= self.meanX
        Xpre /= self.stdX
        kpre1 = covSE(par, self.X, Xpre)
        # Eq2.25
        mean_Ypre = np.dot(np.dot(kpre1.T, self.invKs), self.y)
        temp = self.meany * np.ones(Npre)
        mean_Ypre += temp
        # Eq2.26
        var_Ypre = par[0]**2 - np.diag(np.dot(np.dot(kpre1.T, self.invKs), kpre1))
        # print(Xpre)
        return mean_Ypre, var_Ypre

    # Method for computing the derivative of the negloglike
    def der_negloglik(self, par):
        par = np.squeeze(par)
        K, Ks, invKs = choleInvKs(par, self.X, covSE)
        der = np.zeros(len(par))
        # Eq.5.9 of RW book
        alpha = np.dot(invKs, self.y)
        alpha2 = np.outer(alpha, alpha)
        temp = alpha2 - invKs
        # Derivative w.r.t par[0]
        der_temp = covSE(par, self.X, self.X, trainmode=1)
        der[0] = 1 / 2 * np.sum(temp * der_temp)
        # Derivative w.r.t par[1]
        der_temp1 = covSE(par, self.X, self.X, trainmode=2)
        der[1] = 1 / 2 * np.sum(temp * der_temp1)
        # Derivative w.r.t par[2]
        der[2] = par[2] * np.trace(temp)
        return der
