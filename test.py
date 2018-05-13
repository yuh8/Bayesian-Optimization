import numpy as np
import scipy as sp


A = np.array([[2, 2, 1], [2, 2, 1]])
B = np.ones(3)

print(A.flat)
if len(B.shape) == 1:
    B = B.reshape(1, -1)
print(B.shape[0])
