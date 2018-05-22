import pandas as pd
import numpy as np
from bo_core import (AcqOptimizer, GP)
import matplotlib.pyplot as pl
import os

df = pd.read_csv('./Data/vvt_diesel.csv')

# df.drop(df[df['COV'] > 3].index, inplace=True)

df.dropna(how='any', inplace=True)
df = df.reset_index(drop=True)

N = 200
T = df.shape[0]
idx = np.random.randint(0, T, N)
# Remember to never standardize the input data
# They are automatically standardized in the GP class
Data = df.iloc[idx, :-1].values

print(df)
# angle_min = df.loc[:, [2]].min().values[0]
# angle_max = df.loc[:, [2]].max().values
# VVT_min = df.loc[:, [3]].min().values
# VVT_max = df.loc[:, [3]].max().values
# xbound = ((angle_min, angle_max), (VVT_min, VVT_max))
# print(xbound)
