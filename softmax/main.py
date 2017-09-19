import numpy as np
import math

def softmax(L):
    scores = np.array(list(map(lambda x: math.exp(x), L)))
    total_score = scores.sum()
    return scores / total_score
