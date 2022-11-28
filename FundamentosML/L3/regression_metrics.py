import numpy as np

def mean_absolute_error(x_real, x_pred):
    return (1 / len(x_real)) * np.sum(np.abs(x_real - x_pred))

def mean_squared_error(x_real, x_pred):
    return (1 / len(x_real)) * np.sum(np.square(x_real - x_pred))

def root_mean_squared_error(x_real, x_pred):
    return np.sqrt((1 / len(x_real)) * np.sum(np.square(x_real - x_pred)))

def mean(x):
    return np.sum(x) / len(x)

def variance(x):
    return np.sqrt(np.sum(np.square(x - mean(x))))    

def covariance(x1, x2):
    return np.sum((x1 - mean(x1)) * (x2 - mean(x2)))

def coef_det(x_real, x_pred):
    return covariance(x_real, x_pred) / (variance(x_real) * variance(x_pred))  