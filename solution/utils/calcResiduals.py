import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import inv
from scipy.spatial.transform import Rotation
from scipy.linalg import logm
# self define
from utils.se3 import se3Exp, se3Log
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

    return residuals.flatten(), weights.flatten()


def calculateJacobinResidual(kf_pose_ref, kf_pose, kf_transform_ij, delta):
    t_i, q_i = kf_pose_ref[1:4], kf_pose_ref[4:]
    R_i = Rotation.from_quat(q_i).as_matrix()
    t_j, q_j = kf_pose[1:4], kf_pose[4:]
    R_j = Rotation.from_quat(q_j).as_matrix()
    T_i, T_j = np.zeros(shape=(4,4)), np.zeros(shape=(4,4))
    T_i[3,3], T_j[3,3] = 1, 1
    T_i[:3, :3], T_i[:3, 3] = R_i, t_i
    T_j[:3, :3], T_j[:3, 3] = R_j, t_j

    jacobian = np.zeros(shape=(6, 12))
    eps = 1e-6
    residual = se3Log(np.linalg.inv(kf_transform_ij) @ np.linalg.inv(T_i) @ T_j)
    # devrate for the first variable i
    for k in np.arange(6):
        delta_xi_i, delta_xi_j = delta.copy(), delta.copy()
        delta_xi_i[k] = eps
        residual_k = se3Log(np.linalg.inv(kf_transform_ij) @ np.linalg.inv(T_i) @ se3Exp(- delta_xi_i) @ se3Exp(delta_xi_j) @ T_j)
        jacobian[:, k] = (residual_k - residual) / eps
    # devrate for the second variable j
    for k in np.arange(6):
        delta_xi_i, delta_xi_j = delta.copy(), delta.copy()
        delta_xi_j[k] = eps
        residual_k = se3Log(np.linalg.inv(kf_transform_ij) @ np.linalg.inv(T_i) @ se3Exp(- delta_xi_i) @ se3Exp(delta_xi_j) @ T_j)
        jacobian[:, k+6] = (residual_k - residual) / eps

    return residual, jacobian
