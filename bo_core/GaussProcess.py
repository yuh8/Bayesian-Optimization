
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
        self.X, self.meanX, self.stdX = demean(X)

        # Output data is a 1D numpy array
        y = np.asarray(y, dtype=float)
        self.y, self.meany = demean(y, std=0)
        # Check training data consistency
        n2 = len(self.y)
        if self.N != n2:
            raise ValueError('the number of output and input training data samples must be equal')

    # Compute marginal log-likelihood
    def negloglik(self, par):
        par = np.squeeze(par)
        _, Ks, invKs = choleInvKs(par, self.X, covSE)
        # Compute negative log-likelihood Eq 2.30 of RW book
        negloglik = self.N / 2 * np.log(2 * np.pi)
        negloglik += 1 / 2 * np.dot(np.dot(self.y.T, invKs), self.y)
        negloglik += 1 / 2 * np.log(np.linalg.det(Ks))
        return negloglik

    # method for computing the derivative of the negloglike
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
        der[0] = 1 / 2 * np.trace(np.dot(temp, der_temp))
        # Derivative w.r.t par[1]
        der_temp = covSE(par, self.X, self.X, trainmode=2)
        der[1] = 1 / 2 * np.trace(np.dot(temp, der_temp))
        # Derivative w.r.t par[2]
        der_temp = 2 * par[2] * np.eye(self.N)
        der[2] = 1 / 2 * np.trace(np.dot(temp, der_temp))
        return der

    def fit(self, nstarts=10):
        temp = np.zeros((nstarts, 3))
        fval = np.zeros(nstarts)
        for i in range(0, nstarts):
            par0 = np.random.randn(3)
            res = spmin(self.negloglik, par0, method='L-BFGS-B', jac=self.der_negloglik, options={'gtol': 1e-6, 'disp': False})
            temp[i, :] = np.squeeze(res.x)
            fval[i] = np.squeeze(res.fun)
        idx = np.argmin(fval)
        par_bar = temp[idx, :]
        return par_bar

    # Posterior prediction
    def GP_predict(self, par_bar, Xpre):
        # Testing data sample size
        if len(Xpre.shape) == 1:
            Xpre = Xpre.reshape(1, -1)
        Npre = np.size(Xpre, 0)
        # Demean test data
        temp = np.tile(self.meanX, (Npre, 1))
        Xpre -= temp
        Xpre /= self.stdX
        mean_Ypre, var_Ypre = Predict(par_bar, self.X, self.y, self.meany, Xpre, covSE)
        return mean_Ypre, var_Ypre
