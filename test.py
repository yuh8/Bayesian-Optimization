import numpy as np
from bo_core import *
import matplotlib.pyplot as pl


def out(X):
    return np.cos(X)


N = 10
n = 100
s = 0.005
x0 = np.linspace(-5, 5, N).reshape(N, 1)
y0 = out(x0) + s * np.random.randn(N, 1)

xtest = np.linspace(-5, 5, n).reshape(n, 1)
ytest = out(xtest)
gp = GP(x0, y0)
par = gp.fit(nstarts=20)
print(par)
ypre, varypre = gp.GP_predict(par, xtest)
ypre = np.squeeze(ypre)
print(ypre)
# BO = BayesOpt(x0, y0, ((-10, 10)))

pl.plot(x0, y0, 'bs', ms=8)

pl.gca().fill_between(np.squeeze(xtest), ypre - 2 * np.sqrt(varypre), ypre + 2 * np.sqrt(varypre), color="#dddddd")
pl.plot(xtest, ypre, 'r--', lw=2)
# # pl.axis([-5, 5, -3, 3])
# # pl.title('Three samples from the GP posterior')
pl.show()
