import numpy as np

def get(a : str):
    return{
        "sigmoid" : sigmoid,
        "relu" : relu,
        "tanh" : tanh
    }.get(a, identity)

def identity(x : float):
    return x
def sigmoid(x : float):
    return 1 / (1 + np.exp(-x))

def relu(x : float):
    return np.maximum(0,x)

def tanh(x : float):
    return np.tanh(x)