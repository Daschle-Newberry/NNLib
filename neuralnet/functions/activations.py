import numpy as np
from scipy.special import expit

def get(a : str):
    return{
        "sigmoid" : sigmoid,
        "relu" : relu,
        "tanh" : tanh,
        "None" : none
    }.get(a)

def get_deriv(a : str):
    return {
        "sigmoid": sigmoid_deriv,
        "relu": relu_deriv,
        "tanh": tanh_deriv,
        "None" : none_deriv
    }.get(a)

def none(x : float):
    return x

def none_deriv(x : float):
    return 1

def sigmoid(x : np.ndarray):
    sig = expit(x)
    return sig

def sigmoid_deriv(x : np.ndarray):
    return x * (1 - x)

def relu(x : float):
    return np.maximum(0,x)

def relu_deriv(x : float):
    return x > 0

def tanh(x : float):
    return np.tanh(x)

def tanh_deriv(x : float):
    return 1 - (np.tanh(x) ** 2)
