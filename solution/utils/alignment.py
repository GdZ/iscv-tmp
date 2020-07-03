import numpy as np
from utils.ImageUtils import downscale
from utils.deriveResiduals import deriveResidualsNumeric
from utils.se3 import se3Exp
from utils.se3 import se3Log


def do_alignment(c1, d1, c2, d2, K):
    use_hubernorm = True
    norm_param = 1e100
    if use_hubernorm:
        norm_param = 0.2
    # else:
    #     norm_param = 0.2
    #     norm_param = 1e100

    # % set to zero to use Geman-McClure norm
    # % the initialization of pose
    xi = np.array([[0, 0, 0, 0, 0, 0]]).T

    irefs = []
    drefs = []
    kls = []
    result = []

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
        for i in np.arange(10):
            # % ENABLE ME FOR NUMERIC DERIVATIVES
            Jac, residuals, weights = deriveResidualsNumeric(IRef, DRef, I, xi, Klvl, norm_param, use_hubernorm)
            # % set rows with NaN to 0 (e.g. because out-of-bounds or invalid depth).
            notValid = np.isnan(np.sum(Jac, axis=1) + residuals)
            residuals[notValid] = 0
            Jac[notValid, :] = 0
            weights[notValid] = 0
            vals.append({'Jac': Jac, 'residuals': residuals, 'weights': weights})

            # % do Gauss-Newton step
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
            print('xi: {}\nerr: {}'.format(xi, err))
            result.append({'xi': xi, 'err': err, 'vals': vals})

    return result
