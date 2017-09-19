import numpy as np
import csv
from perceptron_step import perceptronStep

# read in data
with open('data.csv') as csvFile:
    data = csv.reader(csvFile, delimiter = ',')
    X = []
    y = []
    for row in data:
        X.append([float(row[0]), float(row[1])])
        y.append(int(row[2]))
    X = np.array(X)
    y = np.array(y)
    y = np.tile(y, (1, 1)).T
    W = np.array([[1.0],[2.0]])
    b = 1.0
    print(perceptronStep(X, y, W, b))

def stepFunction(t):
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])

def perceptronStep(X, y, W, b, learn_rate = 0.01):
    flags = (np.dot(X, W) + b >= 0) != y
    factors = np.zeros((len(flags), 1))
    factors[np.logical_and(flags, y == 1)] = 1
    factors[np.logical_and(flags, y == 0)] = -1
    W += np.tile(learn_rate * sum(X * np.tile(factors, (1, 2))), (1, 1)).T
    b += learn_rate * sum(factors)
    return W, b

def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines

# define initial state
learn_rate = 0.01
np.random.seed(42)
W = np.array(np.random.rand(2, 1))
x1_max = max(X.T[0])
b = np.random.rand(1)[0] + x1_max

W, b = perceptronStep(X, y, W, b, learn_rate)
print(W, b)
