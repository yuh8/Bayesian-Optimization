import numpy as np
import os

__all__ = ['demean', 'covSE', 'choleInvKs', 'logdetX', 'createFolder']


def demean(X, std=1):
    meanX = np.mean(X, axis=0)
    stdX = np.std(X, axis=0)

    if std == 1:
        X -= meanX
        X /= stdX
        return X, meanX, stdX
    else:
        X -= meanX
        return X, meanX


# Squared exponential kernel and hyperparameter derivatives
def covSE(par, X, Xpre, trainmode=0):
    # Make sure the shape of input matrices are correct
    if len(X.shape) == 1:
        X = X.reshape(1, -1)
    elif len(Xpre.shape) == 1:
        Xpre = Xpre.reshape(1, -1)

    N = X.shape[0]
    Nt = Xpre.shape[0]
    D = X.shape[1]
    exp_temp = np.zeros((N, Nt))

    for i in range(0, N):
        for j in range(0, Nt):
            temp1 = X[i, :] - Xpre[j, :]
            exp_temp[i, j] = np.dot(temp1, temp1)
    # None derivative mode
    if trainmode == 0:
        K = par[0]**2 * np.exp(-1 / 2 * exp_temp / par[1]**2)
    # Der alpha
    elif trainmode == 1:
        K = 2 * par[0] * np.exp(-1 / 2 * exp_temp / par[1]**2)
    # Der l
    elif trainmode == 2:
        K = par[0]**2 * np.exp(-1 / 2 * exp_temp / par[1]**2)
        temp = exp_temp / par[1]**3
        K = K * temp
    # der k/ der x
    else:
        acq_temp = np.zeros((N, D))
        for i in range(0, N):
            acq_temp[i, :] = X[i, :] - Xpre[0, :]
        K = par[0]**2 * np.exp(-1 / 2 * exp_temp[:, 0] / par[1]**2)
        K = np.tile(K, (D, 1)).T
        K = K * acq_temp
    return K


# Stable inversion of symmetric PD matrix
def choleInvKs(par, X, covFcn):
    K = covFcn(par, X, X)
    N = X.shape[0]
    Ks = K + (par[-1]**2 + 1e-6) * np.eye(N)
    # Stable inversion of Ks using cholesky decomposition
    try:
        L = np.linalg.cholesky(Ks)
    except np.linalg.LinAlgError as e:
        print(e)
    else:
        invKs = np.dot(np.linalg.inv(L.T), np.dot(np.linalg.inv(L), np.eye(N)))
        return K, Ks, invKs


# Stable computation of log determinant of large matrix
def logdetX(X):
    try:
        L = np.linalg.cholesky(X)
    except np.linalg.LinAlgError as e:
        print(e)
    else:
        y = 2 * sum(np.log(np.diag(L)))
        return y


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)
