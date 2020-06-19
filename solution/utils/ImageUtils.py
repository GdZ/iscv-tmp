# -*- coding: utf-8 -*-
import numpy as np
from skimage import io
from scipy.linalg import expm
from scipy.linalg import logm
from matplotlib import pyplot as plt
# from scipy.interpolate import interp2d
import numpy.matlib
from scipy import interpolate


def imreadbw(fname):
    print(fname)
    return io.imread(fname, as_gray=True)


"""
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

        xsI, ysI = I.shape
        idxI, idyI = np.arange(xsI, step=2), np.arange(ysI, step=2)
        scaleI = I[idxI, :][:, idyI] + I[1 + idxI, :][:, idyI] + I[idxI, :][:, 1 + idyI] + I[1 + idxI, :][:, 1 + idyI]
        Id = scaleI * .25

        xsD, ysD = D.shape
        idxD, idyD = np.arange(xsD, step=2), np.arange(ysD, step=2)
        DdCountValid = np.sign(D[idxD, :][:, idyD]) + \
                       np.sign(D[1 + idxD, :][:, idyD]) + \
                       np.sign(D[idxD, :][:, 1 + idyD]) + \
                       np.sign(D[1 + idxD, :][:, 1 + idyD])
        index = np.arange(DdCountValid.flatten().shape[0])
        ids = DdCountValid.flatten() != 0
        scaleD = D[idxD, :][:, idyD] + D[1 + idxD, :][:, idyD] + D[idxD, :][:, 1 + idyD] + D[1 + idxD, :][:, 1 + idyD]
        Dd = np.zeros_like(DdCountValid.flatten())
        Dd[index[ids]] = scaleD.flatten()[index[ids]] / DdCountValid.flatten()[index[ids]]
        Dd[index[DdCountValid.flatten() == 0]] = 0
        Dd = Dd.reshape(DdCountValid.shape)

        Id, Dd, Kd = downscale(Id, Dd, Kd, level - 1)
    return Id, Dd, Kd
"""


def downscale(I, D, K, level):
    if level < 1:
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
        xiPerm = se3Log(se3Exp(epsVec) * se3Exp(xi))
        r, w = calcResiduals(IRef, DRef, I, xiPerm, K, norm_param, use_hubernorm)
        Jac[:, j] = (r - residuals) / eps

    return Jac, residuals, weights


def calcResiduals(IRef, DRef, I, xi, K, norm_param, use_hubernorm):
    resdual, weights = [], []
    T = se3Exp(xi)
    R = T[:3, :3]
    t = T[:3, 3]
    KInv = np.linalg.inv(K)
    xImg = np.zeros_like(IRef) - 10
    yImg = np.zeros_like(IRef) - 10

    for x in np.arange(IRef.shape[1]):
        for y in np.arange(IRef.shape[0]):
            p = np.dot(DRef[y, x] * KInv, np.array([[x - 1], [y - 1], [1]]))
            pTrans = K.dot(R.dot(p) + t.reshape(p.shape))
            if pTrans[2] > 0 and DRef[y, x] > 0:
                xImg[y, x] = pTrans[0] / pTrans[2] + 1.
                yImg[y, x] = pTrans[1] / pTrans[2] + 1.

    # residuals = IRef - interp2d(I, xImg, yImg)
    f = interpolate.interp2d(np.real(xImg), np.real(yImg), I)
    residuals = IRef - f(pTrans[0] / pTrans[2] + 1, pTrans[1] / pTrans[2] + 1)

    weights = 0 * residuals + 1
    if use_hubernorm:
        idx = np.abs(residuals) > norm_param
        weights[idx] = norm_param / np.abs(residuals[idx])
    else:
        weights = 2. / (1. + residuals ** 2 / norm_param ** 2) ** 2

    # plot residual
    # plt.subplot(1, 2, 1)
    # plt.imshow(residuals, cmap='gray')
    # plot weight
    # plt.subplot(1, 2, 2)
    # plt.imshow(weights, cmap='gray')
    # plt.legend()
    # plt.show()

    return residuals.reshape(I.flatten().shape), weights.reshape(I.flatten().shape)


def alignment(input_dir, rgbs, depths):
    # % first pair of input frames
    K = np.array([[517.3, 0, 318.6], [0, 516.5, 255.3], [0, 0, 1]])
    # %c2 = double(imreadbw('rgb/1305031102.175304_broken.png'));
    c2 = np.double(imreadbw('{}/{}'.format(input_dir, rgbs[0])))
    c1 = np.double(imreadbw('{}/{}'.format(input_dir, rgbs[1])))
    d2 = np.double(imreadbw('{}/{}'.format(input_dir, depths[0]))) / 5000
    d1 = np.double(imreadbw('{}/{}'.format(input_dir, depths[1]))) / 5000
    # % result:
    # % approximately  -0.0018    0.0065    0.0369   -0.0287   -0.0184   -0.0004
    # % set to zero to use Geman-McClure norm
    use_hubernorm = True
    norm_param = 1e100
    if use_hubernorm:
        norm_param = 0.2
    else:
        norm_param = 0.2
    norm_param = 1e100

    # % initialization
    xi = np.array([[0, 0, 0, 0, 0, 0]]).T

    irefs = []
    drefs = []
    kls = []
    errors = []
    # % pyramid levels
    for i, lvl in enumerate(np.arange(5, 1, -1)):
        IRef, DRef, Klvl = downscale(c1, d1, K, lvl)
        I, D, Kl = downscale(c2, d2, K, lvl)
        irefs.append([IRef, I])
        drefs.append([DRef, D])
        kls.append([Klvl, Kl])
        # just do at most 20 steps
        errLast = 1e10
        vals = []
        fig = plt.figure()
        for i in np.arange(10):
            # % ENABLE ME FOR NUMERIC DERIVATIVES
            Jac, residuals, weights = deriveResidualsNumeric(IRef, DRef, I, xi, Klvl, norm_param, use_hubernorm)
            # % set rows with NaN to 0 (e.g. because out-of-bounds or invalid depth).
            notValid = np.isnan(np.sum(Jac, axis=1) + residuals)
            residuals[notValid] = 0
            Jac[notValid, :] = 0
            weights[notValid] = 0
            plt.subplot(1, 2, 1)
            plt.imshow(residuals.reshape(IRef.shape), cmap='gray')
            plt.subplot(1, 2, 2)
            plt.imshow(weights.reshape(IRef.shape), cmap='gray')
            plt.show()
            vals.append({'Jac': Jac, 'residuals': residuals, 'weights': weights})
        # % do Gauss-Newton step
        # upd = np.dot(np.dot(- np.linalg.inv(np.dot(Jac.T, np.multiply(np.matlib.repmat(weights, 6, 1).T, Jac))), Jac.T),
        #              (weights * residuals))
        weights6 = np.matlib.repmat(weights.reshape(weights.flatten().size, 1), 1, 6)
        mat = Jac.T.dot(np.multiply(weights6, Jac))
        if np.linalg.det(mat) != 0:
            inv = - np.linalg.inv(mat)
        else:
            inv = -np.linalg.pinv(mat)
        upd = inv.dot(Jac.T).dot(np.multiply(weights, residuals).reshape(weights.flatten().size, 1))
        lastXi = xi
        xi = se3Log(se3Exp(upd) * se3Exp(xi))
        err = np.mean(residuals * residuals)
        if err / errLast > .995:
            break
        errLast = err
        errors.append({'err': err, 'vals': vals})
    np.save({'irefs': irefs, 'drefs': drefs, 'kls': kls, 'errors': errors})
    return irefs, drefs, kls, errors


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
