import numpy as np
import csv
from perceptron_step import perceptronStep

def stepFunction(t):
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])

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
    W = np.array([[1.0],[1.0]])
    x_min, x_max = min(X.T[0]), max(X.T[0])
    b = -2.0
    boundary_lines = []
    for i in range(25):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, 0.05)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))

import matplotlib.pyplot as plt
flags = y.astype(bool)
plt.plot(X[flags.T[0], 0], X[flags.T[0], 1], 'ro')
plt.plot(X[flags.T[0] == False, 0], X[flags.T[0] == False, 1], 'bo')
plt.title('A vectorized algorithm of the perceptron')
plt.xlabel('X1')
plt.ylabel('X2')
x_min, x_max = min(X.T[0]), max(X.T[0])
x_range = np.linspace(x_min, x_max, 20)
for i in range(5, len(boundary_lines)):
    y_range = x_range * boundary_lines[i][0][0] - boundary_lines[i][0][0]
    if i == len(boundary_lines) - 1:
        plt.plot(x_range, y_range, 'k-')
    else:
        plt.plot(x_range, y_range, 'y--')
plt.show()
