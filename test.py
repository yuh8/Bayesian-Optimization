import numpy as np


A = 2
B = np.ones(3)


s = np.argsort(A)[::-1]
for i in range(0, 10):
    A = np.vstack((A, i))

print(A)
