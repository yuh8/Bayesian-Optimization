import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from bo_core import *
df_train = pd.read_csv('sarcos_train.csv', header=None)
df_test = pd.read_csv('sarcos_test.csv', header=None)
# Training data
xtrain = df_train.loc[:300, :20]
ytrain = df_train.loc[:300, 21]
# Testing data
xtest = df_test.loc[0:300, :20]
ytest = df_test.loc[0:300, 21]
Nt = xtest.shape[0]

gp = GP(xtrain, ytrain)
par = gp.fit(nstarts=20)
ypre, varypre = gp.predict(par, xtest)
ypre = np.squeeze(ypre)

pl.plot(np.arange(0, Nt), ytest, 'b-.', lw=2)
pl.gca().fill_between(np.arange(0, Nt), ypre - 2 * np.sqrt(varypre), ypre + 2 * np.sqrt(varypre), color="#dddddd")
pl.plot(np.arange(0, Nt), ypre, 'r--', lw=2)
pl.show()
