import numpy as np
from utils.calcResiduals import calcResiduals
from utils.se3 import se3Exp
from utils.se3 import se3Log


def deriveResidualsAnalytic(IRef, DRef, I, xi, K, norm_param, use_hubernorm):
    Jac, residual, weights = [], [], []
    T = se3Exp(xi)
    R = T[:3, :3]
    t = T[:3, 4]
    KInv = K ** (-1)
    RKInv = R * K ** (-1)

    xImg = np.zeros_like(IRef) - 10
    yImg = np.zeros_like(IRef) - 10
    xp, yp, zp = np.zeros_like(IRef), np.zeros_like(IRef), np.zeros_like(IRef)
    wxp, wyp, wzp = np.zeros_like(IRef), np.zeros_like(IRef), np.zeros_like(IRef)

    for x in np.arange(IRef.shape[1]):
        for y in np.arange(IRef.shape[0]):
            p = DRef[y, x] * KInv * np.array([[x - 1], [y - 1], 1])
            pTrans = R * p + t
            if pTrans[2] > 0 and DRef[y, x] > 0:
                pTransProj = K * pTrans
                xImg[y, x] = pTransProj[0] / pTransProj[2]
                yImg[y, x] = pTransProj[1] / pTransProj[2]
                xp, yp, zp = p
                wxp, wyp, wzp = pTrans

    # ========= calculate actual derivative. ===============
    # 1.: calculate image derivatives, and interpolate at warped positions.
    dxI, dyI = np.zeros_like(I), np.zeros_like(I)

    # % 2.: get warped 3d points (x', y', z').

    # % 3. Jacobian calculation

    return Jac, residual, weights


def deriveResidualsNumeric(IRef, DRef, I, xi, K, norm_param, use_hubernorm):
    # Jac, residual, weights = [], [], []
    eps = 1e-6
    Jac = np.zeros(shape=(I.flatten().shape[0], 6))
    residuals, weights = calcResiduals(IRef, DRef, I, xi, K,
                                       norm_param, use_hubernorm)
    for j in np.arange(6):
        epsVec = np.zeros([6, 1])
        epsVec[j] = eps
        xiPerm = se3Log(se3Exp(epsVec) @ se3Exp(xi))
        r, w = calcResiduals(IRef, DRef, I, xiPerm, K, norm_param, use_hubernorm)
        Jac[:, j] = (r - residuals) / eps

    return Jac, residuals, weights