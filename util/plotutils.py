import numpy as np
from matplotlib import  axes

from NNLib.models.sequential import Sequential

def plot_2d_func(x : np.ndarray, y: np.ndarray, ax : axes):
    ax.plot(x,y)

def plot_2d_network_func(x : np.ndarray, network : Sequential, ax : axes):
    y = []
    for i in x:
        y.append(network.predict(i))

    ax.plot(x,y)

def plot_2d_scatter(xy : np.ndarray, z : np.ndarray, ax : axes):
    x,y = xy.T

    x_range = [x.min() - .5 * x.std(),x.max() + .5 * x.std()]
    y_range = [y.min() - .5 * y.std(),y.max() + .5 * y.std()]

    z = z.reshape(-1)
    x0 = xy[z == 0]

    x1 = xy[z == 1]

    x0 = np.array(x0)
    x1 = np.array(x1)

    ax.axis(x_range + y_range)

    ax.scatter(x0[:, 0], x0[:, 1], marker="o", c="b")
    ax.scatter(x1[:, 0], x1[:, 1], marker="x", c="r")

def plot_2d_heatmap(xy : np.ndarray,network, ax : axes):
    x, y = xy.T

    x_lin = np.linspace(x.min() - .5 * x.std(), x.max() + .5 * x.std(),300)
    y_lin = np.linspace(y.min() - .5 * y.std(), y.max() + .5 * y.std(),300)

    X,Y = np.meshgrid(x_lin, y_lin)


    coords = np.stack([X.ravel(), Y.ravel()], axis = 1)


    Z = network.predict(coords)


    Z = Z.reshape(X.shape)
    ax.contourf(X,Y,Z, levels = 100, cmap = "inferno")


