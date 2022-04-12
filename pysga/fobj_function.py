# coding: utf-8
import numpy as np


# Example function
def fobj(x):
    y = np.zeros((x.shape[0], 1))
    s1 = np.zeros(x.shape[0])
    s2 = np.zeros(x.shape[0])
    for i in range(5):
        n = i + 1
        s1 = s1 + n * np.cos((n + 1) * x[:, 0] + n)
        s2 = s2 + n * np.cos((n + 1) * x[:, 1] + n)
    y[:, 0] = s1 * s2
    return y
