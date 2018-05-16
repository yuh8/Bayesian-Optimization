import pandas as pd
import numpy as np
from bo_core import (AcqOptimizer, GP)
import matplotlib.pyplot as pl

df = pd.read_csv('VVT.csv', header=None)
df.dropna(how='all', inplace=True)
df = df.reset_index(drop=True)
idx = np.random.randint(0, 1000, 100)
xtrain0 = np.asarray(df.loc[idx, :3])
param = np.load('param.npy').item()

A = [1, 2, 3, 4]
print(['{}'.format(i) for i in range(15, 75, 5)])
columns = ['{}'.format(i) for i in range(1000, 6200, 200)]
index = ['{}'.format(i) for i in range(15, 75, 5)]
df_angle = pd.DataFrame(index=index, columns=columns, dtype=float)
df_VVT = pd.DataFrame(index=index, columns=columns, dtype=float)
df_angle['1000'] = 10
i = 1000
print(df_angle['{}'.format(i)])
# angle_min = df.loc[:, [2]].min().values[0]
# angle_max = df.loc[:, [2]].max().values
# VVT_min = df.loc[:, [3]].min().values
# VVT_max = df.loc[:, [3]].max().values
# xbound = ((angle_min, angle_max), (VVT_min, VVT_max))
# print(xbound)
