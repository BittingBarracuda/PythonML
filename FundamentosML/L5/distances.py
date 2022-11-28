import numpy as np

def euclidean(x, Y):
    diff_sq = np.square(x - Y)
    return np.sqrt(np.sum(diff_sq, axis = 1))

def manhattan(x, Y):
    diff = np.abs(x - Y)
    return np.sum(diff, axis = 1)

def chess(x, Y):
    diff = np.abs(x - Y)
    return np.max(diff, axis = 1)