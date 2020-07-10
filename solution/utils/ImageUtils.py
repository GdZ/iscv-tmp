# -*- coding: utf-8 -*-
import numpy as np
from skimage import io
from utils.debug import logD
from utils.debug import logV


def imReadByGray(file_path):
    logV(file_path)
    return io.imread(file_path, as_gray=True)


def downscale(o_img, o_depth, K, scaled_level):
    """
    Args:
        o_img: original input rgb-image
        o_depth: original input depth-image
        K: calibration matrix under scale_level
        scaled_level: the scale level
    Returns:
        scaled_img, scaled_depth, scaled_k
    """
    if scaled_level <= 1:
        scaled_img = o_img
        scaled_depth = o_depth
        scaled_k = K
        return scaled_img, scaled_depth, scaled_k

    else:
        # compute new scaled k
        scaled_k = np.array([[K[0][0] / 2,           0, (K[0][2] + 0.5) / 2 - 0.5],
                             [          0, K[1][1] / 2, (K[1][2] + 0.5) / 2 - 0.5],
                             [          0,           0,                         1]])

        # compute scaled rgb-image
        scaled_img = (o_img[0::2, 0::2] + o_img[1::2, 0::2] + o_img[0::2, 1::2] + o_img[1::2, 1::2]) * 0.25

        # compute scaled depth-image
        valid_depth_count = np.sign(o_depth[0::2, 0::2]) + np.sign(o_depth[1::2, 0::2]) + np.sign(o_depth[0::2, 1::2]) + np.sign(o_depth[1::2, 1::2])
        merge_depth = o_depth[0::2, 0::2] + o_depth[1::2, 0::2] + o_depth[0::2, 1::2] + o_depth[1::2, 1::2]
        # scaled_depth = np.divide(scaled_depth, valid_depth_count)
        # scaled_depth[np.isnan(scaled_depth)] = 0
        idx = np.arange(valid_depth_count.flatten().shape[0])
        valid_idx = valid_depth_count.flatten() != 0
        scaled_depth = np.zeros_like(valid_depth_count.flatten())
        scaled_depth[idx[valid_idx]] = merge_depth.flatten()[idx[valid_idx]] / valid_depth_count.flatten()[idx[valid_idx]]
        scaled_depth[idx[valid_depth_count.flatten() == 0]] = 0
        scaled_depth = scaled_depth.reshape(valid_depth_count.shape)

        return downscale(scaled_img, scaled_depth, scaled_k, scaled_level - 1)
