
import numpy as np
from scipy.optimize import minimize as spmin


class GaussProcess:
    '''
    Make sure that the columns before the last column
    contain the input data x and the last column
    contains the output y
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
        X, self.meanX = self.demean(X)
        # Decorrelate train data
        self.X, self.varX, self.V = self.decorInputData(X)
        # Demean test data
        temp = np.tile(self.meanX, (self.Npre, 1))
        Xpre -= temp.T
        # Decorrelate test data
        self.Xpre = np.dot(self.V.T, Xpre)

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

    @staticmethod
    def decorInputData(X):
        N = np.size(X, 1)
        meanX = np.mean(X, axis=1)
        temp = np.tile(meanX, (N, 1))
        X -= temp.T
        # Compute covariance matrix
        cov = 1 / (N - 1) * np.dot(X.T, X)
        w, V = np.linalg.eigh(cov)
        deco_X = np.dot(V.T, X)
        return deco_X, w, V

    # Squared exponential kernel
    @staticmethod
    def covSE(par, X, xtest, trainmode=False):
        n1 = np.size(X, 1)
        n2 = np.size(xtest, 1)
        K = np.zeros((n1, n2))
        for i in range(0, n1):
            for j in range(0, n2):
                temp = np.divide(X[:, i], np.sqrt(par[1])) - np.divide(xtest[:, j], np.sqrt(par[1]))
                if not trainmode:
                    K[i, j] = np.square(par) * np.exp(-np.dot(temp, temp) / 2)
                else:
                    K[i, j] = 2 * par * np.exp(-np.dot(temp, temp) / 2)
        return K

    def choleInvCov(self, alpha, sigma):
        K = self.covSE(par[0], self.X, self.X, self.varX)
        Ks = K + np.square(par[1]) * np.eye(self.N)
        # Stable inversion of Ks using cholesky decomposition
        L = np.linalg.cholesky(Ks)
        invKs = np.inv(L.T) * (np.inv(L) * np.eye(self.N))
        return Ks

    # Compute marginal log-likelihood
    def negloglik(self, par):
        K = self.covSE(par[0], self.X, self.X, self.varX)
        Ks = K + np.square(par[1]) * np.eye(self.N)
        # Stable inversion of Ks using cholesky decomposition
        L = np.linalg.cholesky(Ks)
        invKs = np.inv(L.T) * (np.inv(L) * np.eye(self.N))
        # Compute negative log-likelihood
        negloglik = self.N / 2 * np.log(2 * np.pi)
        negloglik += 1 / 2 * np.dot(self.y * invKs, self.y)
        negloglik += 1 / 2 * np.log(np.linalg.det(Ks))
        return negloglik

    # method for computing the derivative of the negloglike
    def der_negloglik(self, par):
        K = self.covSE(par[0], self.X, self.X, self.varX)
        Ks = K + np.square(par[1]) * np.eye(self.N)
        # Stable inversion of K using cholesky decomposition
        L = np.linalg.cholesky(Ks)
        invKs = np.inv(L.T) * (np.inv(L) * np.eye(self.N))
        der = np.zeros(2)
        # Eq.5.9 of RW book
        alpha = np.dot(invKs, self.y)
        alpha2 = np.outer(alpha, alpha)
        der[0] = self.covSE(par[0], self.X, self.X, self.varX, trainmode=True)
        der[0] = 1 / 2 * np.trace((alpha2 - invKs) * der[0])
        der[1] = K + 2 * par[1] * np.eye(self.N)
        der[1] = 1 / 2 * np.trace((alpha2 - invKs) * der[1])
        return der

    def minimize(self):
        par0 = np.array([0.01, 0.01])
        par_bar = spmin(self.negloglik, par0, method='BFGS', jac=self.der_negloglik, options={'disp': True})
        return par_bar

    def Predict(self, par):
        K = self.covSE(par[0], self.X, self.X, self.varX)
        Ks = K + par[1]**2 * np.eye(self.N)
        L = np.linalg.cholesky(Ks)
        invKs = np.inv(L.T) * (np.inv(L) * np.eye(self.N))
        kpre = self.covSE(par[0], self.X, self.Xpre, self.varX)
        mean_Ypre = np.dot(kpre.T,)
