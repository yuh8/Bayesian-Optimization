
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
from bo_core import GP
'''
This module connects the algorithm to external enviroment
'''

# User import initial training data, the number of data samples must be at least 2
df = pd.read_csv('VVT.csv', header=None)
df.dropna(how='all', inplace=True)
df = df.reset_index(drop=True)

# Setting up training examples
N = 100
Nt = 500
Ntstart = 400
Ntend = Ntstart + Nt - 1
alpha = 0.6
idx = np.random.randint(0, 800, N)
# Remember to never standardize the input data
# They are automatically standardized in the GP class
Xtrain = df.loc[idx, :3].values
ytrain1 = df.loc[idx, 5].values
ytrain2 = df.loc[idx, 4].values
ytest1 = df.loc[Ntstart:Ntend, 5].values
ytest2 = df.loc[Ntstart:Ntend, 4].values

# Standardize output data such that they
# are scale compatible with multiobjective programming
meany1 = np.mean(ytrain1)
stdy1 = np.std(ytrain1)
meany2 = np.mean(ytrain2)
stdy2 = np.std(ytrain2)
ytrain = (ytrain1 - meany1) / stdy1
# * alpha - (ytrain2 - meany2) / stdy2 * (1 - alpha)
xtest = df.loc[Ntstart:Ntend, :3].values
ytest = (ytest1 - meany1) / stdy1
# * alpha - (ytest2 - meany2) / stdy2 * (1 - alpha)


# Initiate automatic hand-shake
gp = GP(Xtrain, ytrain)
par_bar, meanX, stdX = gp.fit(nstarts=20)
# define bound
angle_min = df.loc[:, [2]].min().values
angle_max = df.loc[:, [2]].max().values
VVT_min = df.loc[:, [3]].min().values
VVT_max = df.loc[:, [3]].max().values
xbound = ((*angle_min, *angle_max), (*VVT_min, *VVT_max))
param = {}
param['par_bar'] = par_bar
param['meany1'] = meany1
param['stdy1'] = stdy1
param['xbound'] = xbound
param['gp'] = gp
np.save('param.npy', param)
ypre, varypre = gp.predict(par_bar, xtest)
ypre = np.squeeze(ypre)
pl.plot(np.arange(0, Nt), ytest, 'b-.', lw=2)
pl.gca().fill_between(np.arange(0, Nt), ypre - 2 * np.sqrt(varypre), ypre + 2 * np.sqrt(varypre), color="#dddddd")
pl.plot(np.arange(0, Nt), ypre, 'r--', lw=2)
pl.show()
