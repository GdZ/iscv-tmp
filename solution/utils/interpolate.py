import numpy as np


def interp2d(o, idx, idy):
    """
    Args:
        o: original matrix, which need to interpolate by linear
        idx: index range of x-axis
        idy: index range of y-axis

    Returns:
        the interpolated matrix
    """
    idx_arr = np.asarray(idx)
    idy_arr = np.asarray(idy)
    o_arr = np.asarray(o)

    # here must use floor to get approximation
    idx0 = np.floor(idx_arr).astype(np.int)
    idx1 = idx0 + 1
    idy0 = np.floor(idy_arr).astype(np.int)
    idy1 = idy0 + 1

    r, c = np.array(o_arr.shape) - 1
    clip_idx0 = np.clip(idx0, 0, c)
    clip_idx1 = np.clip(idx1, 0, c)
    clip_idy0 = np.clip(idy0, 0, r)
    clip_idy1 = np.clip(idy1, 0, r)

    Ia = o_arr[clip_idy0, clip_idx0]
    Ib = o_arr[clip_idy1, clip_idx0]
    Ic = o_arr[clip_idy0, clip_idx1]
    Id = o_arr[clip_idy1, clip_idx1]

    wa = (clip_idx1 - idx_arr) * (clip_idy1 - idy_arr)
    wb = (clip_idx1 - idx_arr) * (idy_arr - clip_idy0)
    wc = (idx_arr - clip_idx0) * (clip_idy1 - idy_arr)
    wd = (idx_arr - clip_idx0) * (idy_arr - clip_idy0)

    tmp = (wa * Ia + wb * Ib + wc * Ic + wd * Id).astype(np.float)
    tmp[idx_arr < 0] = np.nan
    tmp[idy_arr < 0] = np.nan
    tmp[idx_arr > c] = np.nan
    tmp[idy_arr > r] = np.nan

    return tmp
