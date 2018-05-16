import numpy as np
import pandas as pd
from bo_core import AcqOptimizer

# load fitted paramters
param = np.load('param_ft.npy').item()
par_bar = param['par_bar']
xbound = param['xbound']
gp = param['gp']

# Creat dataframe for storing data
columns = ['{}'.format(i) for i in range(1000, 6200, 200)]
index = ['{}'.format(i) for i in range(15, 75, 5)]
df_angle = pd.DataFrame(index=index, columns=columns, dtype=float)
df_VVT = pd.DataFrame(index=index, columns=columns, dtype=float)

# Running calibration
for i in range(15, 75, 5):
    for j in range(1000, 6200, 200):
        Acq = AcqOptimizer(par_bar, gp, xbound, [], [j, i])
        xbest, _ = Acq.optim(nstarts=20, mode=1)
        df_angle.loc['{}'.format(i), '{}'.format(j)] = xbest[0]
        df_VVT.loc['{}'.format(i), '{}'.format(j)] = xbest[1]
df_VVT.to_csv('VVTFuelcalibr.csv', sep=',')
df_angle.to_csv('angleFuelcalibr.csv', sep=',')

# # Plot
# import matplotlib.pyplot as plt
# import mpl_toolkits.mplot3d
# from matplotlib import cm
# N = 40
# X = np.linspace(xbound[0][0], xbound[0][1], N)
# Y = np.linspace(xbound[1][0], xbound[1][1], N)

# X, Y = np.meshgrid(X, Y)
# X = X.reshape(N**2)
# Y = Y.reshape(N**2)

# z = np.zeros(N**2)
# for i in range(0, N**2):
#     z[i], _ = gp.predict(par_bar, np.hstack((test_point, [X[i], Y[i]])))
# # fig = plt.figure(figsize=plt.figaspect(0.5))
# ax = fig.add_subplot(1, 2, 1, projection='3d')
# ax.plot_trisurf(X, Y, z, cmap=cm.jet, linewidth=0.1)
# ax.set_xlabel('angle')
# ax.set_ylabel('VVT')
# ax.set_zlabel('fuel')

# fig.add_subplot(1, 2, 2)
# X = X.reshape(N, N)
# Y = Y.reshape(N, N)
# Z = z.reshape(N, N)
# CS = plt.contour(X, Y, Z)
# plt.plot(X.item(np.argmin(Z)), Y.item(np.argmin(Z)), 'r+')
# plt.clabel(CS, inline=1, fontsize=6)
# plt.xlabel('angle')
# plt.ylabel('VVT')
# plt.show()
