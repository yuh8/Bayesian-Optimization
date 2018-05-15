import pandas as pd
import numpy as np
from bo_core import (AcqOptimizer, GP)
df = pd.read_csv('VVT.csv', header=None)
df.dropna(how='all', inplace=True)
df = df.reset_index(drop=True)
idx = np.random.randint(0, 1000, 100)
xtrain0 = df.loc[idx, :3]
ytrain0 = df.loc[10, 5] * 0.6 - df.loc[10, 4] * 0.4


angle_min = df.loc[:, [2]].min().values[0]
angle_max = df.loc[:, [2]].max().values
VVT_min = df.loc[:, [3]].min().values
VVT_max = df.loc[:, [3]].max().values
xbound = ((angle_min, angle_max), (VVT_min, VVT_max))
print(xbound)
