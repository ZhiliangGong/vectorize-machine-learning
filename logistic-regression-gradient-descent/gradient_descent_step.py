import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gradient_descent_step(X, y, W, b, learn_rate = 0.01):
    pred = sigmoid(np.dot(X, W))
    error = -y * np.log(pred) - (1-y) * np.log(1 - pred)
    W -= -np.dot((y - pred).T, X).T
    b -= -np.dot((y-pred).T, np.ones((y.shape[0], 1)))[0][0]
    return W, b, error.sum()
