import numpy as np

def get(a : str):
    return{
        "sigmoid" : sigmoid,
        "relu" : relu,
        "tanh" : tanh
    }.get(a, identity)
def get_deriv(a : str):
    return {
        "sigmoid": sigmoid_deriv,
        "relu": relu,
        "tanh": tanh
    }.get(a, identity)

def identity(x : float):
    return x

def sigmoid(x : float):
    return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x : np.ndarray):
    return x * (1 - x)

def relu(x : float):
    return np.maximum(0,x)
def tanh(x : float):
    return np.tanh(x)