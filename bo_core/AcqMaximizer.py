import numpy as np
from scipy.optimize import minimize as spmin
from scipy.stats import norm


class AcqMax:

    def __init__(self, par, gp, xbound, ybest, xconst=[], method='UCB'):
        # GaussProcess Parameters
        self.par = par
        # Sequence of tuples in which each tuple contains the min and max of a specific variable
        self.bound = xbound
        # The best output corresponding to the best input
        self.ybest = ybest
        # constant speed and load for engine caliberation
        self.xconst = xconst
        # Gaussian process object
        self.gp = gp
        self.tol = 1e-10
        self.method = method

    def ExpImp(self, x, flag=False):
        '''
        x should be in its row form, non-standardized!!!
        '''
        temp = x.tolist()
        m_x, s_x = self.gp.predict(self.par, np.array(temp))
        eta = np.sqrt(s_x)
        u = (m_x - self.ybest) / eta
        EI = eta * (u * norm.cdf(u) + norm.pdf(u))
        if flag:
            return EI, m_x, s_x
        else:
            return -1 * EI

    def UCB(self, x, flag=False):
        '''
        x should be in its row form, non-standardized!!!
        '''
        # Convert to list to avoid object being updated in class
        temp = x.tolist()
        kappa = 2
        m_x, s_x = self.gp.predict(self.par, np.array(temp))
        EI = m_x + kappa * np.sqrt(s_x)
        if flag:
            return EI, m_x, s_x
        else:
            return -1 * EI

    # Simply searching for min of mean function
    def min_mx(self, x):
        '''
        x and xconst should be in their row form
        '''
        m_x, _ = self.gp.predict(self.par, np.hstack((self.xconst, x)))
        return m_x

    def optim(self, nstarts=30, mode=0):
        # Single mode
        if mode == 0:
            if self.method.lower() == 'ucb':
                objfun = self.UCB
            if self.method.lower() == 'ei':
                objfun = self.ExpImp
            if np.size(self.bound[0]) > 1:
                d = len(self.bound)
                bound = self.bound
            else:
                d = 1
                bound = (self.bound,)
            min_fun = np.inf
            for i in range(0, nstarts):
                par0 = np.random.uniform(self.bound[0], self.bound[1], d)

                res = spmin(objfun, par0, method='L-BFGS-B', bounds=bound, options={'maxfun': 100, 'disp': False})

                # Parsimonious trick for finding the min
                if res.fun < min_fun:
                    min_fun = res.fun
                    x = res.x
            res = spmin(objfun, x, method='L-BFGS-B', bounds=bound, options={'gtol': self.tol, 'disp': False})
            xbest_new = res.x.tolist()
            ybest_new, _ = self.gp.predict(self.par, res.x)
            return xbest_new, ybest_new
        # Batch mode for engine calibration
        elif mode == 1:
            d = len(self.bound)
            min_fun = np.inf
            for i in range(0, nstarts):
                par0 = np.random.uniform(self.bound[0], self.bound[1], d)
                res = spmin(self.min_mx, par0, method='L-BFGS-B', bounds=self.bound, options={'gtol': self.tol, 'disp': False})
                # Parsimonious trick for finding the max
                if res.fun < min_fun:
                    min_fun = res.fun
                    xbest_new = res.x.tolist()
            ybest_new, _ = self.gp.predict(self.par, np.hstack((self.xconst, res.x)))
            return xbest_new, ybest_new
