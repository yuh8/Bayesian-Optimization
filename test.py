import numpy as np
import scipy as sp
from bo_core import GaussProcess

a = np.array([1, 2, 3, 4])
s = np.diag(a * 2)
print(np.dot(s, a.T))
