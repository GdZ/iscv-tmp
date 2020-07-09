import numpy as np
from numpy import matlib
from scipy.linalg import cho_factor, cho_solve
from utils.ImageUtils import downscale
from utils.deriveResiduals import deriveResidualsNumeric
from utils.se3 import se3Exp
from utils.se3 import se3Log
from utils.debug import isDebug
from utils.debug import logD


def doAlignment(ref_img, ref_depth, t_img, t_depth, k):
    use_hubernorm = True
    norm_param = 1e100
    if use_hubernorm:
        norm_param = 0.2
    else:
        norm_param = 0.2
    norm_param = 1e100

    # % set to zero to use Geman-McClure norm
    # % the initialization of pose
    xi = np.array([[0, 0, 0, 0, 0, 0]]).T
    xi_arr = []
    err_arr = []
    residuals = []
    weights = []

    # % pyramid levels
    upper, lower, delta = 5, 4, -1
    for i, scaled_level in enumerate(np.arange(upper, lower, delta)):
        scaled_ref_img, scaled_ref_depth, scaled_rk = downscale(ref_img, ref_depth, k, scaled_level)
        scaled_img, scaled_depth, scaled_k = downscale(t_img, t_depth, k, scaled_level)

        # just do at most 20 steps
        N = 20
        err_last = 1e10
        for j in np.arange(N):
            # % ENABLE ME FOR NUMERIC DERIVATIVES
            Jac, residual, weight = deriveResidualsNumeric(scaled_ref_img, scaled_ref_depth,
                                                             scaled_img, xi, scaled_rk,
                                                             norm_param, use_hubernorm)

            # % set rows with NaN to 0 (e.g. because out-of-bounds or invalid depth).
            not_valid = np.isnan(np.sum(Jac, axis=1) + residual)
            residual[not_valid] = 0
            Jac[not_valid, :] = 0
            weight[not_valid] = 0

            # % do Gauss-Newton step
            weights6 = matlib.repmat(weight.reshape(weight.flatten().size, 1), 1, 6)
            mat = Jac.T.dot(np.multiply(weights6, Jac))
            if np.linalg.det(mat) != 0:
                # normal inverse
                # inv = - np.linalg(mat)
                # use cho_factor to solve the inverse of the matrix
                c, low = cho_factor(mat)
                inv = cho_solve((c, low), np.eye(mat.shape[0]))
            else:
                inv = np.linalg.pinv(mat)
            upd = - inv.dot(Jac.T).dot(np.multiply(weight, residual).reshape(weight.flatten().size, 1))

            last_xi = xi
            xi = se3Log(se3Exp(upd) @ se3Exp(xi))
            err = np.mean(residual * residual)
            xi_arr.append(xi)
            err_arr.append(err)
            logD('level: {}, error: {:.8f}, xi: {}'.format(scaled_level, err, xi))
            if err / err_last > .995:
                break
            err_last = err
        residuals.append(residual)
        weights.append(weight)

    return xi_arr, err_arr
