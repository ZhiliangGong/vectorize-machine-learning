import csv
import numpy as np
import random
from gradient_descent_step import gradient_descent_step
from visualize import visualize

with open('data.csv') as csvFile:
    rows = csv.reader(csvFile, delimiter = ',')
    mat = []
    for row in rows:
        mat.append(list(map(lambda x: float(x), row)))
    mat = np.array(mat)
    X = mat[:, 0 : 2]
    y = np.tile(mat[:, 2], (1, 1)).T

    W = np.array(np.random.rand(2,1))*2 - 1
    b = np.random.rand(1)[0]*2 - 1

    learn_rate = 0.01
    lines = []
    for i in range(100):
        W, b, error = gradient_descent_step(X, y, W, b, learn_rate)
        lines.append([-W[0][0] / -W[1][0], -b/W[1][0]])
    visualize(X, y, lines)
