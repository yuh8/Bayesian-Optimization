import numpy as np
import scipy as sp

from bo_core import fitGP

X = np.array([1, 2, 3, 4], dtype=float)
s = fitGP(X)
new_X = s.loglik()
print(new_X)
# print(np.mean(new_X))
# print(np.var(new_X))
