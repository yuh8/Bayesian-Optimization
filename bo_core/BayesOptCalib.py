import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .Optimizer import AcqOptimizer
from .GaussProcess import GP
from .Functions import createFolder


class BOCalib:
    """class for training the calibrator"""

    def __init__(self, Data, AngleRange=None, VVTRange=None, speedRange=None, loadRange=None, doulbeCalib=False, alpha=None):
        # Input is a dataframe
        Data = np.asarray(Data, dtype=float)
        # Number of training data samples
        self.N = np.around(0.7 * Data.shape[0]).astype(int)
        # Number of testing data samples
        self.Nt = Data.shape[0] - self.N
        # Training input dataset
        self.Xtrain = Data[:self.N, :4]
        # Testing input dataset
        self.Xtest = Data[self.N:, :4]
        # Standardize output data for multiobjective optimization
        ytrain = Data[:self.N, -2:]
        ytest = Data[self.N:, -2:]
        meany = np.mean(ytrain, axis=0)
        stdy = np.std(ytrain, axis=0)
        ytrain -= meany
        ytrain /= stdy
        ytest -= meany
        ytest /= stdy
        # If multiobjective optimization
        # Considering both fuel and torque
        if doulbeCalib:
            if alpha is None:
                raise ValueError('Weight of fuel consumption must be provided')
            else:
                self.ytrain = ytrain[:, 0] * (1 - alpha) + ytrain[:, 1] * alpha
                self.ytest = ytest[:, 0] * (1 - alpha) + ytest[:, 1] * alpha
        # Considering only fuel
        else:
            self.ytrain = ytrain[:, 1]
            self.ytest = ytest[:, 1]
        self.bound = (AngleRange, VVTRange)
        self.speedRange = speedRange
        self.loadRange = loadRange

    def fitGP(self, nstarts=20, plot=False):
        gp = GP(self.Xtrain, self.ytrain)
        par_bar, _, _ = gp.fit(nstarts)
        ypre, varypre = gp.predict(par_bar, self.Xtest)
        if plot:
            ybar = np.mean(self.ytest)
            S_tot = np.sum((self.ytest - ybar)**2)
            S_res = np.sum((self.ytest - ypre)**2)
            R2 = 1 - S_res / S_tot
            plt.plot(np.arange(0, self.Nt), self.ytest, 'b-.', lw=2, label='real')
            plt.gca().fill_between(np.arange(0, self.Nt), ypre - 2 * np.sqrt(varypre), ypre + 2 * np.sqrt(varypre), color="#dddddd")
            plt.plot(np.arange(0, self.Nt), ypre, 'r--', lw=2, label='prediction with R2 = {}'.format(R2))
            plt.legend()
            plt.show()
        return gp, par_bar

    @property
    def generateDOE(self):
        loadRange = range(self.loadRange[0], self.loadRange[1] + self.loadRange[2], self.loadRange[2])
        speedRange = range(self.speedRange[0], self.speedRange[1] + self.speedRange[2], self.speedRange[2])
        columns = ['{}'.format(i) for i in speedRange]
        index = ['{}'.format(i) for i in loadRange]
        df_angle = pd.DataFrame(index=index, columns=columns, dtype=float)
        df_VVT = pd.DataFrame(index=index, columns=columns, dtype=float)
        gp, par_bar = self.fitGP(plot=False)
        count = 0
        for i in loadRange:
            for j in speedRange:
                count += 1
                self.saveFig(gp, par_bar, [j, i], count)
                Acq = AcqOptimizer(par_bar, gp, self.bound, [], [j, i])
                xbest, _ = Acq.optim(nstarts=5, mode=1)
                df_angle.loc['{}'.format(i), '{}'.format(j)] = xbest[0]
                df_VVT.loc['{}'.format(i), '{}'.format(j)] = xbest[1]
        createFolder('./Results/')
        df_VVT.to_csv('Results/VVTCalibr.csv', sep=',')
        df_angle.to_csv('Results/AngleCalibr.csv', sep=',')

    def saveFig(self, gp, par_bar, test_point, count):
        N = 40
        X = np.linspace(self.bound[0][0], self.bound[0][1], N)
        Y = np.linspace(self.bound[1][0], self.bound[1][1], N)

        X, Y = np.meshgrid(X, Y)
        X = X.reshape(N**2)
        Y = Y.reshape(N**2)

        z = np.zeros(N**2)
        for i in range(0, N**2):
            z[i], _ = gp.predict(par_bar, np.hstack((test_point, [X[i], Y[i]])))

        X = X.reshape(N, N)
        Y = Y.reshape(N, N)
        Z = z.reshape(N, N)
        plt.figure()
        plt.contourf(X, Y, Z, 25, cmap=plt.cm.jet)
        plt.colorbar()
        plt.plot(X.item(np.argmin(Z)), Y.item(np.argmin(Z)), 'r+', ms=20)
        plt.xlabel('angle')
        plt.ylabel('VVT')
        plt.title('Speed = {} rpm, load = {} %'.format(test_point[0], test_point[1]), fontsize=16)
        createFolder('./Figs/')
        plt.savefig('Figs/Fig{}.png'.format(count))
        plt.close()
