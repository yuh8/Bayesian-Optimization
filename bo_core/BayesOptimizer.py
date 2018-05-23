import numpy as np
from .AcqMaximizer import AcqMax
from .GaussProcess import GP


class BayesOpt:
    """
    Bayesian Optimization Algorithm
    X, y are the augumented data samples from previous exploration
    xbest,ybest are the augmented bests from previous exploitation
    """

    def __init__(self, X, y, ybest, xbound, xconst=[]):
        # Making sure the size of the inputs are compatible
        if len(X.shape) == 1 or len(y) == 1:
            raise ValueError('number of initial points must be at least 2')

        # Initiating the single BO procedure
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.ybest = ybest
        # The bound for search
        self.bound = xbound

    def max_opts(self, ns=10):
        gp = GP(self.X, self.y)
        par_bar, _, _ = gp.fit(nstarts=ns)
        self.Acq = AcqMax(par_bar, gp, self.bound, self.ybest)
        xbest_new, ybest_new = self.Acq.optim(nstarts=ns, mode=0)
        return xbest_new, ybest_new

    def PredictEI(self, Xtry):
        EI, m_x, s_x = self.Acq.UCB(Xtry, flag=True)
        return EI, m_x, s_x
