import pandas as pd
import numpy as np
from bo_core import (AcqOptimizer, GP)
import matplotlib.pyplot as pl
import os

df = pd.read_csv('VVT.csv', header=None)
df.dropna(how='all', inplace=True)
df = df.reset_index(drop=True)
idx = np.random.randint(0, 1000, 100)
xtrain0 = np.asarray(df.loc[idx, :3])
param = np.load('param.npy').item()

A = np.array([5, 6, 7, 8])
path = os.getcwd()


def Folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


Folder('./Figs/')


# angle_min = df.loc[:, [2]].min().values[0]
# angle_max = df.loc[:, [2]].max().values
# VVT_min = df.loc[:, [3]].min().values
# VVT_max = df.loc[:, [3]].max().values
# xbound = ((angle_min, angle_max), (VVT_min, VVT_max))
# print(xbound)
