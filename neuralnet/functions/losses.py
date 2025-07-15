import numpy as np

def get(a : str):
    return{
        "MSE" : MSE,
        "BCE" : BCE
    }.get(a, None)

def get_deriv(a : str):
    return {
        "MSE" : MSE_deriv,
        "BCE": BCE_deriv

    }.get(a, None)
def MSE(yHat : np.ndarray, y : np.ndarray):
        return np.mean((yHat - y) ** 2)

def MSE_deriv(yHat : np.ndarray, y : np.ndarray):
    return 2 * (yHat - y)

def BCE(yHat : np.ndarray, y : np.ndarray):
    epsilon = 1E-8
    yHat = np.clip(yHat, epsilon,1 - epsilon)
    return np.mean(-y * np.log(yHat) - (1 - y) * np.log(1 - yHat))

def BCE_deriv(yHat : np.ndarray, y : np.ndarray):
    epsilon = 1E-8
    yHat = np.clip(yHat, epsilon, 1 - epsilon)
    return (-y / yHat) + ((1 - y) / (1 - yHat))
