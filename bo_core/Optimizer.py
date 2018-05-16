import numpy as np
from scipy.optimize import minimize as spmin
from scipy.stats import norm


class AcqOptimizer:
    """Bayesian optimization based on expected improvment"""

    def __init__(self, par, gp, xbound, ybest, xconst=[]):
        # GaussProcess Parameters
        self.par = par
        # Sequence of tuples in which each tuple contains the min and max of a specific variable
        self.bound = xbound
        # The best output corresponding to the best input
        self.ybest = ybest
        self.xconst = xconst
        self.gp = gp

    def ExpImprove(self, x):
        '''
        x should be in its row form, non-standardized!!!
        '''
        m_x, s_x = self.gp.predict(self.par, x)
        self.eta = np.sqrt(s_x)
        self.u = (self.ybest - m_x) / self.eta
        EI = self.eta * (self.u * norm.cdf(self.u) + norm.pdf(self.u))
        return EI

    # Simply searching for min of mean function
    def min_mx(self, x):
        '''
        x and xconst should be in their row form
        '''
        m_x, _ = self.gp.predict(self.par, np.hstack((self.xconst, x)))
        return m_x

    def optim(self, nstarts=10, mode=0):
        if mode == 0:
            d = len(self.bound)
            temp = np.zeros((nstarts, d))
            fval = np.zeros(nstarts)
            for i in range(0, nstarts):
                par0 = np.random.uniform(self.bound[0], self.bound[1], d)
                res = spmin(self.ExpImprove, par0, method='L-BFGS-B', bounds=self.bound, options={'xtol': 1e-4, 'disp': False})
                temp[i, :] = np.squeeze(res.x)
                fval[i] = np.squeeze(res.fun)
            idx = np.argmin(fval)
            xbest_new = temp[idx, :]
            ybest_new, _ = self.gp.predict(self.par, xbest_new)
            return xbest_new, ybest_new
        else:
            d = len(self.bound)
            temp = np.zeros((nstarts, d))
            fval = np.zeros(nstarts)
            for i in range(0, nstarts):
                par0 = np.random.uniform(self.bound[0], self.bound[1], d)
                res = spmin(self.min_mx, par0, method='L-BFGS-B', bounds=self.bound, options={'gtol': 1e-4, 'disp': False})
                temp[i, :] = np.squeeze(res.x)
                fval[i] = np.squeeze(res.fun)
            idx = np.argmin(fval)
            xbest_new = temp[idx, :]
            ybest_new, _ = self.gp.predict(self.par, np.hstack((self.xconst, xbest_new)))
            return xbest_new, ybest_new
