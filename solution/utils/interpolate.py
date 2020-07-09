import numpy as np


def bilinear_interpolate(im, yy, xx):
    y = np.asarray(yy)
    x = np.asarray(xx)

    y0 = np.floor(y).astype(int)
    y1 = y0 + 1
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1

    y0 = np.clip(y0, 0, im.shape[1] - 1)
    y1 = np.clip(y1, 0, im.shape[1] - 1)
    x0 = np.clip(x0, 0, im.shape[0] - 1)
    x1 = np.clip(x1, 0, im.shape[0] - 1)

    Ia = im[x0, y0]
    Ib = im[x1, y0]
    Ic = im[x0, y1]
    Id = im[x1, y1]

    wa = (y1 - y) * (x1 - x)
    wb = (y1 - y) * (x - x0)
    wc = (y - y0) * (x1 - x)
    wd = (y - y0) * (x - x0)

    # tmp = np.zeros_like(im)
    tmp = 0
    if y > im.shape[1]-1 or x > im.shape[0]-1 or y < 0 or x < 0:
        tmp = np.nan
    else:
        # tmp = wa * Ia + wb * Ib + wc * Ic + wd * Id
        tmp = np.dot(np.array([[wa, wb, wc, wd]]), np.array([[Ia, Ib, Ic, Id]]).T)
        # print(np.dot(np.array([[wa, wb, wc, wd]]), np.array([[Ia, Ib, Ic, Id]]).T), tmp)

    return tmp
