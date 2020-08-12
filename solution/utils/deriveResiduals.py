import numpy as np
from utils.calcResiduals import calcResiduals
from utils.se3 import se3Exp
from utils.se3 import se3Log


# optim
def deriveResidualsAnalytic(ref_img, ref_depth, img, xi, k, norm_param, use_hubernorm):
    Jac, residual, weights = [], [], []
    T = se3Exp(xi)
    R = T[:3, :3]
    t = T[:3, 4]
    k_inv = k ** (-1)
    rk_inv = R * k ** (-1)

    x_img = np.zeros_like(ref_img) - 10
    y_img = np.zeros_like(ref_img) - 10
    xp, yp, zp = np.zeros_like(ref_img), np.zeros_like(ref_img), np.zeros_like(ref_img)
    wxp, wyp, wzp = np.zeros_like(ref_img), np.zeros_like(ref_img), np.zeros_like(ref_img)

    for x in np.arange(ref_img.shape[1]):
        for y in np.arange(ref_img.shape[0]):
            p = ref_depth[y, x] * k_inv * np.array([[x - 1], [y - 1], 1])
            p_trans = R * p + t
            if p_trans[2] > 0 and ref_depth[y, x] > 0:
                p_trans_projection = k * p_trans
                x_img[y, x] = p_trans_projection[0] / p_trans_projection[2]
                y_img[y, x] = p_trans_projection[1] / p_trans_projection[2]
                xp, yp, zp = p
                wxp, wyp, wzp = p_trans

    # ========= calculate actual derivative. ===============
    # 1.: calculate image derivatives, and interpolate at warped positions.
    dxI, dyI = np.zeros_like(img), np.zeros_like(img)

    # % 2.: get warped 3d points (x', y', z').

    # % 3. Jacobian calculation

    return Jac, residual, weights


def derivePoseGraphResidualsNumeric(IRef, DRef, I, xi, K, norm_param, use_hubernorm, rij):
    # Jac, residual, weights = [], [], []
    eps = 1e-6
    Jac = np.zeros(shape=(I.flatten().shape[0], 6))
    residuals, weights = calcResiduals(IRef, DRef, I, xi, K, norm_param, use_hubernorm)

    for j in np.arange(6):
        epsVec = np.zeros([6, 1])
        epsVec[j] = eps
        xiPerm = se3Log(se3Exp(epsVec) @ se3Exp(xi))
        Jac[:, j] = (rij - residuals) / eps

    return Jac, residuals, weights


def deriveResidualsNumeric(IRef, DRef, I, xi, K, norm_param, use_hubernorm):
    # Jac, residual, weights = [], [], []
    eps = 1e-6
    Jac = np.zeros(shape=(I.flatten().shape[0], 6))
    residuals, weights = calcResiduals(IRef, DRef, I, xi, K, norm_param, use_hubernorm)

    for j in np.arange(6):
        epsVec = np.zeros([6, 1])
        epsVec[j] = eps
        xiPerm = se3Log(se3Exp(epsVec) @ se3Exp(xi))
        r, w = calcResiduals(IRef, DRef, I, xiPerm, K, norm_param, use_hubernorm)
        Jac[:, j] = (r - residuals) / eps

    return Jac, residuals, weights
