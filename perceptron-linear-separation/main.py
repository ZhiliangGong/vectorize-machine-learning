import numpy as np
import csv
import random
import sys
from perceptron_step import perceptronStep
from visualize_steps import visualize_steps

def train_perceptron(learn_rate = 0.01, num_epochs = 20):
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
        W = np.array(np.random.rand(2,1))
        x_min, x_max = min(X.T[0]), max(X.T[0])
        b = np.random.rand(1)[0] + x_max
        boundary_lines = []
        for i in range(num_epochs):
            W, b = perceptronStep(X, y, W, b, learn_rate)
            boundary_lines.append((-W[0]/W[1], -b/W[1]))
        visualize_steps(X, y, boundary_lines)

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        learn_rate = float(sys.argv[1])
        num_epochs = int(sys.argv[2])
    elif len(sys.argv) == 2:
        learn_rate = float(sys.argv[1])
        num_epochs = 20
    else:
        learn_rate = 0.01
        num_epochs = 20
    train_perceptron(learn_rate, num_epochs)
