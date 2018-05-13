import numpy as np
from scipy.optimize import minimize as spmin
from scipy.stats import norm
from .GaussProcess import GP
from .Functions import *


class AcqOptimizer:
    """Bayesian optimization based on expected improvment"""

    def __init__(self, parbar, xbound, xbest, ybest, Xt_old, yt_old):
        # GaussProcess Parameters
        self.par = parbar
        # Bound for input space
        self.bound = xbound
        # The best input setting from previous iteration
        self.xbest = xbest
        # The best output corresponding to the best input
        self.ybest = ybest
        # Training data for GaussProcess
        self.Xold = Xt_old
        self.yold = yt_old

    def ExpImprove(self, x):
        gp = GP(self.Xt_old, self.yt_old)
        m_x, s_x = gp.GP_predict(self.parbar, x)
        eta = np.sqrt(s_x)
        u = (self.ybest - m_x) / eta
        EI = u * eta * norm.cdf(u) + eta * norm.pdf(u)
        return EI

    def der_EI(self, x):
        d = len(x)
        deta_dx = np.zeros(d)
        k = covSE(self.par, self.Xold, x)
