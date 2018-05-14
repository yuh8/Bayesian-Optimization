import numpy as np
from .Optimizer import AcqOptimizer
from .GaussProcess import GP


class BayesOpt:
    """Bayesian Optimization Algorithm"""

    def __init__(self, x0, y0, xbound):
        if len(x0.shape) == 1 or len(y0) == 1:
            raise ValueError('number of initial points must be at least 2')
        # Initiating the BO procedure
        X = np.asarray(x0, dtype=float)
        y = np.asarray(y0, dtype=float)
        # Arrange in descending order since we are minimizing
        idx = np.argsort(y)[::-1]
        X = X[idx, :]
        y = y[idx]
        self.X = X
        self.y = y
        self.xbest = X[-1, :]
        self.ybest = y[-1]
        # The bound for search
        self.bound = xbound

    def max_opts(self):
        output = {'Xbest': self.xbest, 'ybest': self.ybest}
        tol = 1e-2
        ybest = self.ybest
        err = np.inf
        while err > tol:
            gp = GP(self.X, self.y)
            self.par_bar, self.meanX = gp.minimize
            Acq = AcqOptimizer(self.par_bar, self.bound, self.ybest, self.X, self.y, self.meanX)
            xbest_new, ybest_new = Acq.optim
            err = ybest_new - ybest
            ybest = ybest_new
            output['Xbest'] = np.hstack((output['Xbest'], xbest_new))
            output['ybest'] = np.vstack((output['ybest'], ybest_new))
        return output
