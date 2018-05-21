
import numpy as np
import pandas as pd
from bo_core import BOCalib
'''
This module connects the algorithm to external enviroment
'''

# User import initial training data, the number of data samples must be at least 2
df = pd.read_csv('./Data/VVT.csv', header=None)
df.dropna(how='all', inplace=True)
df = df.reset_index(drop=True)

N = 200
idx = np.random.randint(0, 800, N)
# Remember to never standardize the input data
# They are automatically standardized in the GP class
Data = df.loc[idx, :].values

BO = BOCalib(Data, AngleRange=(-3, 57), VVTRange=(-44, 6), speedRange=(1000, 6000), loadRange=(15, 75), doulbeCalib=True, alpha=0.6)

BO.generateDOE
