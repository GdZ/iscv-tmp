import numpy as np
from matplotlib import pyplot as plt
from utils.se3 import se3Exp
from utils.interpolate import interp2d
from utils.debug import isDebug


def calcResiduals(ref_img, ref_depth, img, xi, k, norm_param, use_hubernorm):
    # get shorthands (R, t)
    T = se3Exp(xi)
    R = T[:3, :3]
    t = T[:3, 3]

    k_inv = np.linalg.inv(k)

    # these contain the x,y image coordinates of the respective
    # reference-pixel, transformed & projected into the new image.
    # set to -10 initially, as this will give NaN on interpolation later.
    x_img = np.zeros_like(ref_img) - 10
    y_img = np.zeros_like(ref_img) - 10
    interp_img = np.zeros_like(ref_img)

    for x in np.arange(ref_img.shape[1]):
        for y in np.arange(ref_img.shape[0]):
            # point in reference image. note that the pixel-coordinates of the
            # point (1,1) are actually (0,0).
            p = np.dot(ref_depth[y, x] * k_inv, np.array([[x], [y], [1]]))

            # transform to image (unproject, rotate & translate)
            p_trans = k.dot(R.dot(p) + t.reshape(p.shape))

            # if point is valid (depth > 0), project and save result.
            if p_trans[2] > 0 and ref_depth[y, x] > 0:
                x_img[y, x] = p_trans[0] / p_trans[2]
                y_img[y, x] = p_trans[1] / p_trans[2]

    # calculate actual residual (as matrix).
    interp_img = interp2d(img, x_img, y_img)
    residuals = ref_img - interp_img

    weights = 0 * residuals + 1
    if use_hubernorm:
        idy = np.abs(residuals) > norm_param
        weights[idy] = norm_param / np.abs(residuals[idy])
    else:
        weights = 2. / (1. + residuals ** 2 / norm_param ** 2) ** 2

    # plot residual
    if isDebug():
        # fig = plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(residuals, cmap='Greys')
        plt.xlabel('residuals')
        plt.subplot(1, 2, 2)
        plt.imshow(weights, cmap='Greys')
        plt.xlabel('weights')
        # plt.show()

    return residuals.flatten(), weights.flatten()
