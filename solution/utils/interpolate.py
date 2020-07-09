import numpy as np


def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    im = np.asarray(im)

    x0 = np.ceil(x-1).astype(int)
    x0 = np.clip(x0, 0, im.shape[1]-1)
    x1 = x0 + 1
    x1 = np.clip(x1, 0, im.shape[1]-1)

    y0 = np.ceil(y-1).astype(int)
    y0 = np.clip(y0, 0, im.shape[0]-1)
    y1 = y0 + 1
    y1 = np.clip(y1, 0, im.shape[0]-1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # print(wa)

    tmp = (wa * Ia + wb * Ib + wc * Ic + wd * Id).astype(float)
    tmp[x < 0] = np.nan
    tmp[y < 0] = np.nan
    tmp[x > im.shape[1] - 1] = np.nan
    tmp[y > im.shape[0] - 1] = np.nan
    # print(tmp)
    # tmp[x > im.shape[1] - 1] = np.nan
    # tmp[y > im.shape[0]-1] = np.nan
    # tmp[x < 0] = np.nan
    # tmp[y < 0] = np.nan
    #    if x > im.shape[1]-1 or y > im.shape[0]-1 or x < 0 or y < 0:
    #        tmp = np.nan
    #   else:
    #       tmp = wa * Ia + wb * Ib + wc * Ic + wd * Id

    return tmp

n = [[54.5, 17.041667, 31.993],
     [54.5, 17.083333, 31.911],
     [54.458333, 17.041667, 31.945],
     [54.458333, 17.083333, 31.866],
     ]
x = np.repeat([0,1,2,-5],3).reshape(4,3)
y = np.tile([0,1,2],4).reshape(4,3)


print(bilinear_interpolate(n,y,x))
