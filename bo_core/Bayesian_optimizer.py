
import numpy as np
from scipy.optimize import minimize as spmin


class GaussProcess:
    '''
    Make sure that the columns before the last column
    contain the input data x and the last column
    contains the output y
    '''

    def __init__(self, X, y, Xpre, coding=1):
        # Input data row are features, columns are data samples
        # d by N array

        if coding == 1:
            X = np.asarray(X, dtype=float)
            Xpre = np.asarray(X, dtype=float)
            self.N = np.size(X, 1)
            self.Npre = np.size(Xpre, 1)
            X, self.meanX = self.demean(X)
            self.X, self.varx, self.V = self.decorInputData(X)
            temp = np.tile(self.meanX, (self.Npre, 1))
            Xpre -= temp.T
            self.Xpre = np.dot(self.V.T, Xpre)
        else:
            X = np.asarray(X, dtype=float).T
            Xpre = np.asarray(X, dtype=float).T
            self.N = np.size(X, 1)
            self.Npre = np.size(Xpre, 1)
            X, self.meanX = self.demean(X)
            self.X, self.varx, self.V = self.decorInputData(X)
            temp = np.tile(self.meanX, (self.Npre, 1))
            Xpre -= temp.T
            self.Xpre = np.dot(self.V.T, Xpre)
        # Output data
        y = np.asarray(y, dtype=float)
        self.y, self.meany = self.demean(y)
        n1 = np.size(self.X, 1)
        # Output data is 1D numpy array
        n2 = np.size(self.y, 0)
        if n1 != n2:
            raise ValueError('the number of output and input training data samples should be equal')

    @staticmethod
    def demean(X):
        meanX = np.mean(X, axis=1)
        N = np.size(X, 1)
        temp = np.tile(meanX, (N, 1))
        X -= temp.T
        return X, meanX

    def decorInputData(self, X):
        X = self.demean(X)
        cov = 1 / (self.N - 1) * np.dot(X.T, X)
        w, V = np.linalg.eigh(cov)
        deco_X = np.dot(V.T, self.X)
        return deco_X, w, V

    @staticmethod
    def covSE(x, data, vardata, trainmode=False):
        N = np.size(data, 1)
        K = np.zeros((N, N))
        for i in range(0, N):
            for j in range(0, N):
                temp = np.divide(data[:, i], np.sqrt(vardata)) - np.divide(data[:, j], np.sqrt(vardata))
                if not trainmode:
                    K[i, j] = np.square(x) * np.exp(-np.dot(temp, temp) / 2)
                else:
                    K[i, j] = 2 * x * np.exp(-np.dot(temp, temp) / 2)
        return K

    def negloglik(self, par):
        K = self.covSE(par[0], self.X, self.varx)
        Ks = K + np.square(par[1]) * np.eye(self.N)
        # Stable inversion of K using cholesky decomposition
        L = np.linalg.cholesky(Ks)
        invKs = np.inv(L.T) * (np.inv(L) * np.eye(self.N))
        # Compute negative log-likelihood
        negloglik = self.N / 2 * np.log(2 * np.pi)
        negloglik += 1 / 2 * np.dot(self.y * invKs, self.y)
        negloglik += 1 / 2 * np.log(np.linalg.det(Ks))
        return negloglik

    # method for computing the derivative of the negloglike
    def der_negloglik(self, par):
        K = self.covSE(par[0], self.X, self.varx)
        Ks = K + np.square(par[1]) * np.eye(self.N)
        # Stable inversion of K using cholesky decomposition
        L = np.linalg.cholesky(Ks)
        invKs = np.inv(L.T) * (np.inv(L) * np.eye(self.N))
        der = np.zeros(2)
        # Eq.5.9 of RW book
        alpha = np.dot(invKs, self.y)
        alpha2 = np.outer(alpha, alpha)
        der[0] = self.covSE(par[0], self.X, self.varx, trainmode=True)
        der[0] = 1 / 2 * np.trace((alpha2 - invKs) * der[0])
        der[1] = K + 2 * par[1] * np.eye(self.N)
        der[1] = 1 / 2 * np.trace((alpha2 - invKs) * der[1])

    def minimize(self):
        par0 = np.array([0.1, 0.01])
        par_bar = spmin(self.negloglik, par0, method='BFGS', jac=self.der_negloglik, options={'disp': True})
        return par_bar

    def Predict(self):
        pass
