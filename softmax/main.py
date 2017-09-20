import numpy as np
import math

def softmax(L):
    scores = np.array(list(map(lambda x: math.exp(x), L)))
    return scores / scores.sum()

def cross_entropy(Y, P):
    y = np.array(Y)
    p = np.array(P)
    lk = y * p + (1 - y) * (1 - p)
    return - np.log(lk).sum()
