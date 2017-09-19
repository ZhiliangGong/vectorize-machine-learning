import numpy as np

def perceptronStep(X, y, W, b, learn_rate = 0.01):
    print(X.shape, y.shape, W.shape)
    flags = (np.dot(X, W) + b >= 0) != y
    factors = np.zeros((len(flags), 1))
    factors[np.logical_and(flags, y == 1)] = 1
    factors[np.logical_and(flags, y == 0)] = -1
    W += np.tile(learn_rate * sum(X * np.tile(factors, (1, 2))), (1, 1)).T
    b += learn_rate * sum(factors)
    return W, b
