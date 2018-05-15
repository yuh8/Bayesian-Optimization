import numpy as np
from scipy.optimize import minimize as spmin
from scipy.stats import norm
from .GaussProcess import GP
from .Functions import *


class AcqOptimizer:
    """Bayesian optimization based on expected improvment"""

    def __init__(self, parbar, xbound, ybest, Xt_old, yt_old, meanX, xconst=[]):
        # GaussProcess Parameters
        self.par = parbar
        # Sequence of tuples in which each tuple contains the min and max of a specific variable
        self.bound = xbound
        # The best output corresponding to the best input
        self.ybest = ybest
        # Training data for GaussProcess
        self.Xold = Xt_old
        self.yold = yt_old
        self.meanX = meanX
        self.xconst = xconst
        self.gp = GP(self.Xold, self.yold)

    def ExpImprove(self, x):
        m_x, s_x = self.gp.predict(self.par, x)
        self.eta = np.sqrt(s_x)
        self.u = (self.ybest - m_x) / self.eta
        EI = self.eta * (self.u * norm.cdf(self.u) + norm.pdf(self.u))
        return EI

    def der_EI(self, x):
        _, _, invKs = choleInvKs(self.par, self.Xold, covSE)
        k = covSE(self.par, self.Xold, x)
        dk_dx = covSE(self.par, self.Xold, x, trainmode=3)
        deta_dx = np.dot(np.dot(-dk_dx.T, invKs), k)
        du_dx = -(np.dot(np.dot(dk_dx.T, invKs), self.yold) + self.u * deta_dx) / self.eta
        dEI_dx = deta_dx * (self.u * norm.cdf(self.u) + norm.pdf(self.u)) + self.eta * du_dx * norm.cdf(self.u)
        return dEI_dx

    # Simply searching for min of mean function
    def min_mx(self, x):
        m_x, _ = self.gp.predict(self.par, np.hstack((self.xconst, x)))
        return m_x

    def optim(self, nstarts=10, mode=0):
        if mode == 0:
            d = self.Xold.shape[1]
            temp = np.zeros((nstarts, d))
            fval = np.zeros(nstarts)
            for i in range(0, nstarts):
                par0 = np.random.uniform(self.bound[0], self.bound[1], d)
                res = spmin(self.ExpImprove, par0, method='L-BFGS-B', jac=self.der_EI, bounds=self.bound, options={'xtol': 1e-4, 'disp': False})
                temp[i, :] = np.squeeze(res.x)
                fval[i] = np.squeeze(res.fun)
            idx = np.argmin(fval)
            xbest_new = temp[idx, :]
            xbest_new += self.meanX
            ybest_new, _ = self.gp.predict(self.par, xbest_new)
            return xbest_new, ybest_new
        else:
            d = self.Xold.shape[1] - len(self.xconst)
            temp = np.zeros((nstarts, d))
            fval = np.zeros(nstarts)
            for i in range(0, nstarts):
                par0 = np.random.uniform(self.bound[0], self.bound[1], d)
                res = spmin(self.min_mx, par0, method='L-BFGS-B', bounds=self.bound, options={'gtol': 1e-4, 'disp': False})
                temp[i, :] = np.squeeze(res.x)
                fval[i] = np.squeeze(res.fun)
            idx = np.argmin(fval)
            xbest_new = temp[idx, :]
            xbest_new += self.meanX[-d:]
            ybest_new, _ = self.gp.predict(self.par, np.hstack((self.xconst, xbest_new)))
            return xbest_new, ybest_new
