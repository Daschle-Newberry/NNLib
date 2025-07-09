import numpy as np
from matplotlib import  axes

def plot_2d_scatter(xy : np.ndarray, z : np.ndarray, ax : axes):

    x0 = []

    x1 = []

    x_range = [130,300]
    y_range = [10,16]
    first = True
    for i in range(len(xy)):

        # x_range = [xy[i][0],xy[i][0]] if first else [min(x_range[0],xy[i][0]),max(x_range[1],xy[i][0])]
        # y_range = [xy[i][1],xy[i][1]] if first else [min(y_range[0],xy[i][1]),max(y_range[1],xy[i][1])]

        if(z[i] == 0):
            x0.append(xy[i])
        else:
            x1.append(xy[i])

    print(x_range)

    x0 = np.array(x0)
    x1 = np.array(x1)

    ax.axis(x_range + y_range)

    ax.scatter(x0[:, 0], x0[:, 1], marker="o", c="b")
    ax.scatter(x1[:, 0], x1[:, 1], marker="x", c="r")