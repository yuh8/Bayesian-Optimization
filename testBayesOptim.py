from bo_core import BayesOpt
import numpy as np
import matplotlib.pyplot as plt

# Function to be minimized


def process(x):
    y = np.exp(-(x - 2)**2) + np.exp(-(x - 6)**2 / 10) + 1 / (x**2 + 1)
    return y


# Generate two data points from function
Xtrain = np.array([-1, 8]).reshape(-1, 1)
ytrain = process(Xtrain).reshape(1, -1)[0]
ybest_pre = np.min(ytrain)
Xtrain0 = Xtrain.tolist()
ytrain0 = ytrain.tolist()

# Bayesian optimization iteration
bound = (-2, 10)
temp = np.linspace(-10, 10, 100)
tol = 1e-4
err = np.inf
fig = plt.gcf()
fig.show()
fig.canvas.draw()

itr1 = 0
while itr1 < 30:
    itr1 += 1
    BO = BayesOpt(Xtrain, ytrain, ybest_pre, bound)

    xbest_cur, ybest_cur = BO.max_opts(ns=20)
    # Augment new location for exploration and add results
    Xtrain0.append(xbest_cur)
    Xtrain = np.array(Xtrain0)
    ytrain_new = process(Xtrain[-1])
    ytrain0.append(ytrain_new[0].tolist())
    ytrain = np.array(ytrain0)
    ybest_pre = np.min(ytrain)

    # Demo
    EI, m_x, s_x = BO.PredictEI(np.linspace(-10, 10, 100).reshape(-1, 1))
    # print(EI.argmax(), xbest_cur)
    plt.subplot(211)
    plt.plot(temp, m_x, 'b-')
    plt.plot(temp,process(temp),'r--')
    plt.gca().fill_between(temp, m_x - 2 * np.sqrt(s_x), m_x + 2 * np.sqrt(s_x), color="#dddddd")
    plt.plot(np.array(Xtrain), process(np.array(Xtrain)), 'r*')
    plt.title('Posterior fit')
    plt.xlim([-2, 10])
    plt.subplot(212)
    plt.plot(temp, EI, 'k-')
    plt.plot([xbest_cur, xbest_cur], [0, max(EI)])
    plt.title('Expected Improvement')

    plt.xlim([-2, 10])
    plt.pause(1)  # Pause 1 second

    plt.clf()
    fig.canvas.draw()
