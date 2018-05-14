import numpy as np
from scipy.optimize import minimize as spmin
from scipy.stats import norm
from .GaussProcess import GP
from .Functions import *


class AcqOptimizer:
    """Bayesian optimization based on expected improvment"""

    def __init__(self, parbar, xbound, ybest, Xt_old, yt_old, meanX):
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

    def ExpImprove(self, x):
        gp = GP(self.Xt_old, self.yt_old)
        m_x, s_x = gp.GP_predict(self.parbar, x)
        self.eta = np.sqrt(s_x)
        self.u = (self.ybest - m_x) / self.eta
        EI = self.eta * (self.u * norm.cdf(self.u) + norm.pdf(self.u))
        return EI

    def der_EI(self, x):
        _, _, invKs = choleInvKs(self.par, self.Xold, covSE)
        k = covSE(self.par, self.Xold, x)
        dk_dx = covSE(self.par, self.Xold, x, trainmode=3)
        deta_dx = np.dot(np.dot(-dk_dx.T, invKs), k)
        du_dx = -(np.dot(np.dot(dk_dx.T, invKs), y) + self.u * deta_dx) / self.eta
        dEI_dx = deta_dx * (self.u * norm.cdf(self.u) + norm.pdf(self.u)) + self.eta * du_dx * norm.cdf(self.u)
        return dEI_dx

    @property
    def optim(self):
        d = self.Xt_old.shape[1]
        par0 = 0.1 * np.ones(d)
        xbest_new = spmin(self.ExpImprove, par0, method='L-BFGS-B', jac=self.der_EI, bounds=self.bound, options={'xtol': 1e-6, 'disp': False})
        xbest_new += self.meanX
        ybest_new, _ = gp.GP_predict(self.parbar, xbest_new)
        return xbest_new, ybest_new
