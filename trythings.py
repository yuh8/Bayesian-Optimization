import pandas as pd
import numpy as np
df_train = pd.read_csv('sarcos_train.csv', header=None)
df_test = pd.read_csv('sarcos_test.csv', header=None)
# Training data
xtrain = df_train.loc[:, :20]
ytrain = df_train.loc[:, 21]
# Testing data
xtest = df_test.loc[200:299, :20]
ytest = df_test.loc[200:299, 21]

print(ytrain.shape)
