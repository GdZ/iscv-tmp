# -*- coding: utf-8 -*-
import numpy as np
from skimage import io
from scipy.linalg import expm
from scipy.linalg import logm
from matplotlib import pyplot as plt


def imreadbw(fname):
    return io.imread(fname, as_gray=True)


def downscale(I, D, K, level):
    if level <= 1:
        Id = I
        Dd = D
        Kd = K
    else:
        Kd = np.zeros(shape=(3, 3))
        Kd[0, 0] = K[0, 0] / 2
        Kd[0, 2] = (K[0, 2] + .5) / 2 - .5
        Kd[1, 1] = K[1, 1] / 2
        Kd[1, 2] = (K[1, 2] + .5) / 2 - .5
        Kd[2, :] = [0, 0, 1]

        Id = I

        Dd = D

    return Id, Dd, Kd


def deriveResidualsAnalytic():
    Jac, residual, weights = [], [], []
    return Jac, residual, weights


def deriveResidualsNumeric(IRef, DRef, I, xi, K, norm_param, use_hubernorm):
    Jac, residual, weights = [], [], []
    eps = 1e-6
    Jac = np.zeros(I.shape, 6)
    calcResiduals(IRef=IRef, DRef=DRef, I=I, xi=xi,
                  K=K, norm_param=norm_param, use_hubernorm=use_hubernorm)
    for j in np.arange(6):
        epsVec = np.zeros(6, 1)
        epsVec[j] = eps
        xiPerm = se3Log(se3Exp(epsVec) * se3Exp(xi))
        print(xiPerm)
        pass

    return Jac, residual, weights


def calcResiduals(IRef, DRef, I, xi, K, norm_param, use_hubernorm):
    T = se3Exp(xi)
    R = T[:3, :3]
    t = T[:3, 4]
    KInv = K**(-1)
    xImg = np.zeros_like(IRef) - 10
    yImg = np.zeros_like(IRef) - 10

    for x in np.arange(IRef.shape[1]):
        for y in np.arange(IRef.shape[1]):
            p = DRef[y, x] * KInv * np.array([[x-1], [y-1], [0]])

    return resdual, weights


def downscale(I, D, K, level):
    if level <= 1:
        Id = I
        Dd = D
        Kd = K
    else:
        Kd = np.zeros(shape=(3, 3))
        Kd[0, 0] = K[0, 0] / 2
        Kd[0, 2] = (K[0, 2] + .5) / 2 - .5
        Kd[1, 1] = K[1, 1] / 2
        Kd[1, 2] = (K[1, 2] + .5) / 2 - .5
        Kd[2, :] = [0, 0, 1]

        # Id =

    return (Id, Dd, Kd)


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

