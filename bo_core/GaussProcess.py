
import numpy as np
from scipy.optimize import minimize as spmin


class GP:
    '''
    Fitting and prediction with Gaussian process with sqaure exponential kernel
    '''

    def __init__(self, X, y, Xpre, coding=1):
        # Input data rows are features, columns are data samples
        # d by N array
        if coding == 1:
            X = np.asarray(X, dtype=float)
            Xpre = np.asarray(X, dtype=float)
        else:
            X = np.asarray(X, dtype=float).T
            Xpre = np.asarray(X, dtype=float).T
        # Training data sample size
        self.N = np.size(X, 1)
        # Testing data sample size
        self.Npre = np.size(Xpre, 1)
        # Demean train data
        self.X, self.meanX = self.demean(X)
        # Demean test data
        temp = np.tile(self.meanX, (self.Npre, 1))
        self.Xpre -= temp.T

        # Output data is 1D numpy array
        y = np.asarray(y, dtype=float)
        self.y, self.meany = self.demean(y)
        # Check training data consistency
        n2 = np.size(self.y, 0)
        if self.N != n2:
            raise ValueError('the number of output and input training data samples must be equal')

    @staticmethod
    def demean(X):
        meanX = np.mean(X, axis=1)
        N = np.size(X, 1)
        temp = np.tile(meanX, (N, 1))
        X -= temp.T
        return X, meanX

    # Squared exponential kernel and hyperparameter derivatives
    @staticmethod
    def covSE(par, X, xtest, trainmode=0):
        N = np.size(X, 1)
        Nt = np.size(xtest, 1)
        exp_temp = np.zeros((N, Nt))
        for i in range(0, N):
            for j in range(0, Nt):
                temp = X[:, i] - xtest[:, j]
                exp_temp[i, j] = np.dot(temp, temp)
        if trainmode == 0:
            K = np.square(par[0]) * np.exp(-1 / 2 * exp_temp / np.square(par[1]))
        elif trainmode == 1:
            K = 2 * par[0] * np.exp(-1 / 2 * exp_temp / np.square(par[1]))
        else:
            K = np.square(par[0]) * np.exp(-1 / 2 * exp_temp / np.square(par[1]))
            temp = exp_temp / np.power(par[1], 3)
            K = np.multiply(K, temp)
        return K

    # Stable inversion of symmetric PD matrix
    def choleInvKs(self, par):
        K = self.covSE(par, self.X, self.X)
        Ks = K + np.square(par[2]) * np.eye(self.N)
        # Stable inversion of Ks using cholesky decomposition
        L = np.linalg.cholesky(Ks)
        invKs = np.dot(np.inv(L.T), np.dot(np.inv(L), np.eye(self.N)))
        return K, Ks, invKs

    # Compute marginal log-likelihood
    def negloglik(self, par):
        _, Ks, invKs = self.choleInvKs(par)
        # Compute negative log-likelihood Eq 2.30 of RW book
        negloglik = self.N / 2 * np.log(2 * np.pi)
        negloglik += 1 / 2 * np.dot(np.dot(self.y, invKs), self.y)
        negloglik += 1 / 2 * np.log(np.linalg.det(Ks))
        return negloglik

    # method for computing the derivative of the negloglike
    def der_negloglik(self, par):
        K, Ks, invKs = self.choleInvKs(par)
        der = np.zeros(3)
        # Eq.5.9 of RW book
        alpha = np.dot(invKs, self.y)
        alpha2 = np.outer(alpha, alpha)
        # Derivative w.r.t par[0]
        der[0] = self.covSE(par, self.X, self.X, trainmode=1)
        der[0] = 1 / 2 * np.trace(np.dot(alpha2 - invKs, der[0]))
        # Derivative w.r.t par[1]
        der[1] = self.covSE(par, self.X, self.X, trainmode=2)
        der[1] = 1 / 2 * np.trace(np.dot(alpha2 - invKs, der[1]))
        # Derivative w.r.t par[2]
        der[2] = K + 2 * par[2] * np.eye(self.N)
        der[2] = 1 / 2 * np.trace(np.dot(alpha2 - invKs, der[2]))
        return der

    @property
    def minimize(self):
        par0 = np.array([0.01, 0.01, 0.01])
        par_bar = spmin(self.negloglik, par0, method='BFGS', jac=self.der_negloglik, options={'disp': True})
        return par_bar

    def Predict(self, par):
        # Eq2.25 and 2.26 of RW book
        _, _, invKs = self.choleInvKs(par)
        kpre1 = self.covSE(par, self.X, self.Xpre)
        kpre2 = self.covSE(par, self.Xpre, self.Xpre)
        # Eq2.25
        mean_Ypre = np.dot(np.dot(kpre1.T, invKs), self.y)
        temp = np.tile(self.meany, (self.Npre, 1))
        mean_Ypre += temp
        # Eq2.26
        var_Ypre = kpre2 - np.dot(np.dot(kpre1.T, invKs), kpre1)
        return mean_Ypre, var_Ypre
