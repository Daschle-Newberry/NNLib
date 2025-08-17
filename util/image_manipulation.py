from idlelib.pyparse import trans

import numpy as np


def random_transform_2d(x : np.ndarray, wiggle : float):
    channels, height, width = x.shape
    max_y, max_x = (int(height * wiggle), int(width * wiggle))

    dy = int(np.random.uniform(-max_y,max_y))
    dx = int(np.random.uniform(-max_x,max_x))

    res = transform_2d(x,dy,dx)

    return res



def transform_2d(x : np.ndarray, dy : int, dx : int):
    channels, height, width = x.shape
    padded = np.pad(x,((0,0),(max(-dy,0),max(dy,0)),(max(dx,0),max(-dx,0))))

    cropped = padded[:,max(dy,0):max(0,dy) + height, max(-dx,0):max(0,-dx) + width]

    return cropped

