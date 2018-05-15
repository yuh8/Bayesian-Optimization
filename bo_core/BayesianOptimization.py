import numpy as np
from .Optimizer import AcqOptimizer
from .GaussProcess import GP


class BayesOpt:
    """
    Bayesian Optimization Algorithm
    X, y are the augumented data samples from previous exploration
    xbest,ybest are the augmented bests from previous exploitation
    """

    def __init__(self, X, y, xbest, ybest, xbound, xconst=[]):
        # Making sure the size of the inputs are compatible
        if len(X.shape) == 1 or len(y) == 1:
            raise ValueError('number of initial points must be at least 2')

        # Initiating the single BO procedure
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.X = X
        self.y = y

        if len(xbest.shape) == 1 or len(ybest) == 1:
            self.xbest = xbest.reshape(1, -1)
            self.ybest = ybest
            self.ybestcur = ybest
        else:
            # Arrange the previous bests in descending order since we are minimizing
            idx = np.argsort(ybest)[::-1]
            self.xbest = xbest[idx, :]
            self.ybest = ybest[idx]
            self.ybestcur = ybest[-1]

        # The bound for search
        self.bound = xbound

    def max_opts(self, ns=10):
        output = {'Xbest': self.xbest, 'ybest': self.ybest}
        gp = GP(self.X, self.y)
        par_bar, meanX = gp.fit(nstarts=ns)
        Acq = AcqOptimizer(par_bar, self.bound, self.ybestcur, self.X, self.y, meanX)
        xbest_new, ybest_new = Acq.optim(nstarts=ns)
        output['Xbest'] = np.hstack((output['Xbest'], xbest_new))
        output['ybest'] = np.hstack((output['ybest'], ybest_new))
        # tol = 1e-4
        # ybest = self.ybest
        # err = np.inf
        # while err > tol:
        #     gp = GP(self.X, self.y)
        #     self.par_bar, self.meanX = gp.minimize
        #     Acq = AcqOptimizer(self.par_bar, self.bound, self.ybest, self.X, self.y, self.meanX)
        #     xbest_new, ybest_new = Acq.optim(nstarts=10)
        #     err = ybest_new - ybest
        #     ybest = ybest_new
        #     output['Xbest'] = np.hstack((output['Xbest'], xbest_new))
        #     output['ybest'] = np.vstack((output['ybest'], ybest_new))
        return output
