import numpy as np
from scipy.linalg import expm
from scipy.linalg import logm


def se3Log(x):
    lg = logm(x)
    twist = np.array([lg[0, 3], lg[1, 3], lg[2, 3],
                      lg[2, 1], lg[0, 2], lg[1, 0]]).T
    return twist


def se3Exp(x):
    M = np.array([[0, -x[5], x[4], x[0]],
                  [x[5], 0, -x[3], x[1]],
                  [-x[4], x[3], 0, x[2]],
                  [0, 0, 0, 0]])

    return expm(M)
