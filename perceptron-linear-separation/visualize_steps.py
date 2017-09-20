import numpy as np
import matplotlib.pyplot as plt

def visualize_steps(X, y, boundary_lines):
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
