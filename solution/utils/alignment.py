import numpy as np
from numpy import matlib
from scipy.linalg import cho_factor, cho_solve
from utils.ImageUtils import downscale
from utils.deriveResiduals import deriveResidualsNumeric
from utils.deriveResiduals import derivePoseGraphResidualsNumeric
from utils.se3 import se3Exp
from utils.se3 import se3Log
from utils.debug import isDebug
from utils.debug import logD


def doAlignment(ref_img, ref_depth, target_img, target_depth, k, scaled_level=5):
    use_hubernorm = True
    norm_param = 1e100
    if use_hubernorm:
        norm_param = 0.2
    else:
        norm_param = 0.2

    # set to zero to use Geman-McClure norm
    xi = np.array([[0, 0, 0, 0, 0, 0]]).T

    ref_i, ref_d, ref_k = downscale(ref_img, ref_depth, k, scaled_level)
    t_i, t_d, t_k = downscale(target_img, target_depth, k, scaled_level)

    # just do at most 20 steps
    N = 20
    err_last = 1e10
    for j in np.arange(N):
        # ENABLE ME FOR NUMERIC DERIVATIVES
        Jac, residual, weight = deriveResidualsNumeric(ref_i, ref_d,
                                                       t_i, xi, ref_k,
                                                       norm_param, use_hubernorm)

        # set rows with NaN to 0 (e.g. because out-of-bounds or invalid depth).
        not_valid = np.isnan(np.sum(Jac, axis=1) + residual)
        residual[not_valid] = 0
        Jac[not_valid, :] = 0
        weight[not_valid] = 0

        # do Gauss-Newton step
        weights6 = matlib.repmat(weight.reshape(weight.flatten().size, 1), 1, 6)
        mat = Jac.T.dot(np.multiply(weights6, Jac))
        if np.linalg.det(mat) != 0:
            # use cho_factor to solve the inverse of the matrix
            c, low = cho_factor(mat)
            inv = cho_solve((c, low), np.eye(mat.shape[0]))
        else:
            inv = np.linalg.pinv(mat)
        upd = - inv.dot(Jac.T).dot(np.multiply(weight, residual).reshape(weight.flatten().size, 1))

        last_xi = xi
        xi = se3Log(se3Exp(upd) @ se3Exp(xi))
        err = np.mean(residual * residual)
        logD('level: {}, error: {:.8f}, xi: {}'.format(scaled_level, err, xi))
        if err / err_last > .995:
            break
        err_last = err
    cov = inv
    # use entropy fomula from slide 08 page 46
    H_xi = 0.5 * np.log(np.linalg.det(2 * np.e * np.pi * cov))
    # H_xi = 0.5 * len(last_xi) * (1 + np.log(2 * np.pi)) + 0.5 * (np.log(np.linalg.det(cov)))

    return xi, err, H_xi


def poseGraph(rgbs, depths, i, j, K, rij):
    xi = np.array([[0, 0, 0, 0, 0, 0]]).T
    use_hubernorm = True
    norm_param = 1e100
    if use_hubernorm:
        norm_param = 0.2
    else:
        norm_param = 0.2

    ref_i, ref_d, ref_k = downscale(rgbs[i], depths[i], K, 5)
    scaled_img, scaled_depth, scaled_k = downscale(rgbs[j], depths[j], K, 5)

    for i in np.arange(20):
        Jac, residual, weight = derivePoseGraphResidualsNumeric(IRef=ref_i, DRef=ref_d, I=scaled_img, xi=xi, K=K, norm_param=norm_param, use_hubernorm=use_hubernorm, rij=rij)
        not_valid = np.isnan(np.sum(Jac, axis=1) + residual)
        residual[not_valid] = 0
        Jac[not_valid, :] = 0
        weight[not_valid] = 0

        # do Gauss-Newton step
        weights6 = matlib.repmat(weight.reshape(weight.flatten().size, 1), 1, 6)
        mat = Jac.T.dot(np.multiply(weights6, Jac))
        if np.linalg.det(mat) != 0:
            # use cho_factor to solve the inverse of the matrix
            c, low = cho_factor(mat)
            inv = cho_solve((c, low), np.eye(mat.shape[0]))
        else:
            inv = np.linalg.pinv(mat)
        upd = - inv.dot(Jac.T).dot(np.multiply(weight, residual).reshape(weight.flatten().size, 1))

        last_xi = xi
        xi = se3Log(se3Exp(upd) @ se3Exp(xi))
        err = np.mean(residual * residual)

        if err / err_last > .995:
            break
        err_last = err
    # d.append(delta)

    return xi
