
import numpy as np
import pandas as pd
from bo_core import BOCalib
'''
This module connects the algorithm to external enviroment
'''

df = pd.read_csv('./Data/vvt_diesel.csv')

df.drop(df[df['COV'] > 3].index, inplace=True)

df.dropna(how='any', inplace=True)
df = df.reset_index(drop=True)

N = 100
T = df.shape[0]
idx = np.random.randint(0, T, T)
# Remember to never standardize the input data
# They are automatically standardized in the GP class
Data = df.iloc[idx, :-1].values


BO = BOCalib(Data, AngleRange=(-5, 46), VVTRange=(0, 48), speedRange=(1200, 5600, 400), loadRange=(100, 2400, 100), doulbeCalib=True, alpha=0.6)

# BO.fitGP(nstarts=20, plot=True)
BO.generateDOE
