import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp2d
from utils.se3 import se3Exp


def calcResiduals(IRef, DRef, I, xi, K, norm_param, use_hubernorm):
    resdual, weights = [], []
    # get shorthands (R, t)
    T = se3Exp(xi)
    R = T[:3, :3]
    t = T[:3, 3]

    KInv = np.linalg.inv(K)

    # these contain the x,y image coordinates of the respective
    # reference-pixel, transformed & projected into the new image.
    # set to -10 initially, as this will give NaN on interpolation later.
    xImg = np.zeros_like(IRef) - 10
    yImg = np.zeros_like(IRef) - 10

    for x in np.arange(IRef.shape[1]):
        for y in np.arange(IRef.shape[0]):
            # point in reference image. note that the pixel-coordinates of the
            # point (1,1) are actually (0,0).
            p = np.dot(DRef[y, x] * KInv, np.array([[x], [y], [1]]))

            # transform to image (unproject, rotate & translate)
            pTrans = K.dot(R.dot(p) + t.reshape(p.shape))

            # if point is valid (depth > 0), project and save result.
            if pTrans[2] > 0 and DRef[y, x] > 0:
                xImg[y, x] = pTrans[0] / pTrans[2]
                yImg[y, x] = pTrans[1] / pTrans[2]

    # calculate actual residual (as matrix).
    # residuals = IRef - interp2d(I, xImg, yImg)
    xnew = np.arange(xImg.shape[1])
    ynew = np.arange(yImg.shape[0])

    # f = interp2d(xnew, ynew, I)
    # Inew = f(xImg.flatten(), yImg.flatten())

    # f = interp2d(xImg, yImg, I)
    # Inew = f(xnew, ynew)

    from utils.interpolate import bilinear_interpolate
    # Inew = bilinear_interpolate(I, xnew, ynew)
    Inew = np.zeros_like(IRef)
    for idx in np.arange(IRef.shape[1]):
        for idy in np.arange(IRef.shape[0]):
            Inew[idy, idx] = bilinear_interpolate(I, xImg[idy, idx], yImg[idy, idx])

    residuals = IRef - Inew

    # residuals[xImg == -10] = np.max(residuals)*1.2
    residuals[xImg == -10] = np.nan
    #  print(residuals)

    weights = 0 * residuals + 1
    if use_hubernorm:
        idx = np.abs(residuals) > norm_param
        weights[idx] = norm_param / np.abs(residuals[idx])
    else:
        weights = 2. / (1. + residuals ** 2 / norm_param ** 2) ** 2

    # plot residual
    # not implement
    # fig = plt.figure(figsize=(12, 12))
    # plt.subplot(1, 2, 1)
    # plt.imshow(residuals, cmap='Greys')
    # plt.xlabel('residuals')
    # plt.subplot(1, 2, 2)
    # plt.imshow(weights, cmap='Greys')
    # plt.xlabel('weights')
    # plt.show()

    return residuals.reshape(I.flatten().shape), weights.reshape(I.flatten().shape)
