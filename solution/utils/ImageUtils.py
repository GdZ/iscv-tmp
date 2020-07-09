# -*- coding: utf-8 -*-
import numpy as np
from skimage import io
# from scipy.linalg import expm
# from scipy.linalg import logm
# from matplotlib import pyplot as plt
# from scipy.interpolate import interp2d
# import numpy.matlib
# from scipy import interpolate


def imreadbw(fname):
    print(fname)
    return io.imread(fname, as_gray=True)


def downscale(I, D, K, level):
    if level <= 1:
        Id = I
        Dd = D
        Kd = K
        return Id, Dd, Kd
    else:
        Kd = np.array([
            [K[0][0] / 2, 0, (K[0][2] + 0.5) / 2 - 0.5],
            [0, K[1][1] / 2, (K[1][2] + 0.5) / 2 - 0.5],
            [0, 0, 1]])
        Id = (I[0::2, 0::2] +
              I[1::2, 0::2] +
              I[0::2, 1::2] +
              I[1::2, 1::2]) * 0.25
        DdCountValid = (np.sign(D[0::2, 0::2]) +
                        np.sign(D[1::2, 0::2]) +
                        np.sign(D[0::2, 1::2]) +
                        np.sign(D[1::2, 1::2]))
        scaleD = (D[0::2, 0::2] +
                  D[1::2, 0::2] +
                  D[0::2, 1::2] +
                  D[1::2, 1::2]
                  )
        # Dd = np.divide(Dd, DdCountValid)
        # Dd[np.isnan(Dd)] = 0
        index = np.arange(DdCountValid.flatten().shape[0])
        ids = DdCountValid.flatten() != 0
        Dd = np.zeros_like(DdCountValid.flatten())
        Dd[index[ids]] = scaleD.flatten()[index[ids]] / DdCountValid.flatten()[index[ids]]
        Dd[index[DdCountValid.flatten() == 0]] = 0
        Dd = Dd.reshape(DdCountValid.shape)

        return downscale(Id, Dd, Kd, level - 1)
