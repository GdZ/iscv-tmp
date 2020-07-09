import numpy as np
from numpy import matlib
from scipy.linalg import cho_factor, cho_solve
from utils.ImageUtils import downscale
from utils.deriveResiduals import deriveResidualsNumeric
from utils.se3 import se3Exp
from utils.se3 import se3Log
from utils.debug import is_debug
from utils.debug import logD


def do_alignment(c1, d1, c2, d2, K):
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

    xis = []
    errors = []
    residuals = []
    weights = []

    # % pyramid levels
    for i, lvl in enumerate(np.arange(5, 4, -1)):
        IRef, DRef, Klvl = downscale(c1, d1, K, lvl)
        I, D, Kl = downscale(c2, d2, K, lvl)

        # just do at most 20 steps
        N = 20
        errLast = 1e10
        for j in np.arange(N):
            # % ENABLE ME FOR NUMERIC DERIVATIVES
            Jac, residuals, weights = deriveResidualsNumeric(IRef, DRef, I, xi, Klvl, norm_param, use_hubernorm)

            # % set rows with NaN to 0 (e.g. because out-of-bounds or invalid depth).
            notValid = np.isnan(np.sum(Jac, axis=1) + residuals)
            residuals[notValid] = 0
            Jac[notValid, :] = 0
            weights[notValid] = 0

            # % do Gauss-Newton step
            weights6 = matlib.repmat(weights.reshape(weights.flatten().size, 1), 1, 6)
            mat = Jac.T.dot(np.multiply(weights6, Jac))
            if np.linalg.det(mat) != 0:
                # normal inverse
                # inv = - np.linalg(mat)
                # use cho_factor to solve the inverse of the matrix
                c, low = cho_factor(mat)
                inv = cho_solve((c, low), np.eye(mat.shape[0]))
            else:
                inv = np.linalg.pinv(mat)
            upd = - inv.dot(Jac.T).dot(np.multiply(weights, residuals).reshape(weights.flatten().size, 1))

            lastXi = xi
            xi = se3Log(se3Exp(upd) @ se3Exp(xi))
            err = np.mean(residuals * residuals)
            xis.append(xi)
            errors.append(err)

            logD('level: {}, error: {:.8f}, xi: {}'.format(lvl, err, xi))

            if err / errLast > .995:
                break
            errLast = err

    return xis, errors
