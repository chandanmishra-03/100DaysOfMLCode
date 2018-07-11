import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors


def plot(X,Y,pred_func):
    mins = np.amin(X,0); 
    mins = mins - 0.1*np.abs(mins);
    maxs = np.amax(X,0); 
    maxs = maxs + 0.1*maxs;

    xs,ys = np.meshgrid(np.linspace(mins[0],maxs[0],300), 
            np.linspace(mins[1], maxs[1], 300));


    Z = pred_func(np.c_[xs.flatten(), ys.flatten()]);
    Z = Z.reshape(xs.shape)

    plt.contourf(xs, ys, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=50,
            cmap=colors.ListedColormap(['orange', 'blue']))
    plt.show()
