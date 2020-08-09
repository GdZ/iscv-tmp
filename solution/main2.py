#! -*- coding: utf-8 -*-
# !/bin/env python
import os
import sys
import argparse
import numpy as np
from matplotlib import pyplot as plt

# self defined function
from utils.dataset import loadData
from utils.ImageUtils import imReadByGray
from utils.dataset import saveData
from utils.debug import logD
from utils.debug import logV
from utils.apps import taskAB, taskC, taskD, taskE


def main(argv):
    parser = argparse.ArgumentParser(
        description="team project"
    )
    parser.add_argument('--input-dir',
                        type=str,
                        default='./data',
                        help='path of dataset folder')
    parser.add_argument('--output-dir',
                        type=str,
                        default='./output',
                        help='path to the out directory')
    args = parser.parse_args(argv)

    input_dir = args.input_dir
    output_dir = args.output_dir

    if os.path.isdir(input_dir):
        os.listdir(input_dir)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # load depth, rgb timestamp
    timestamp_rgb, rgb, timestamp_depth, depth = loadData(input_dir)
    logD('timestamp: {}, rgb: {}, depth: {}'.format(timestamp_rgb.shape, rgb.shape, depth.shape))
    alignment(input_dir, output_dir, t1=timestamp_rgb, rgbs=rgb, t2=timestamp_depth, depths=depth)


def alignment(input_dir, output_dir, t1, rgbs, t2, depths):
    """
    :param input_dir:
    :param t1: timestamp of rgb image
    :param rgbs: rgb-images
    :param t2: timestamp of depth images
    :param depths: depth-images
    """
    # actual parameter, which is copy from visiom.tum
    K = np.array([[520.9, 0, 325.1], [0, 521.0, 249.7], [0, 0, 1]])

    # task (a), (b)
    delta_x_array, pose_w2kf_array, distance_array = taskAB(K,
                                                            colors=rgbs,
                                                            depths=depths,
                                                            input_dir=input_dir,
                                                            timestamp_color=t1,
                                                            timestampe_depth=t2
                                                            , batch_size=len(rgbs)
                                                            , threshold=0.052
                                                            )
    np.save('{}/delta_xs_array'.format(output_dir), delta_x_array)
    np.save('{}/pose_w2kf_array'.format(output_dir), pose_w2kf_array)
    np.save('{}/distance_array'.format(output_dir), distance_array)
    saveData(pose_w2kf_array, outdir=input_dir, fn='estimate_ab_{}.txt'.format(.052))

    # task (c)
    keyframe_w2kf_array, entropy_array, kf_idx_array = taskC(K,
                                                             input_dir=input_dir,
                                                             colors=rgbs,
                                                             depths=depths,
                                                             timestamp_color=t1,
                                                             timestampe_depth=t2,
                                                             lower=0.91,  # 0.915
                                                             upper=1.04
                                                             , batch_size=len(rgbs)
                                                             )
    np.save('{}/keyframe_w2kf_array'.format(output_dir), keyframe_w2kf_array)
    np.save('{}/entropy_array'.format(output_dir), entropy_array)
    np.save('{}/kf_idx_array'.format(output_dir), kf_idx_array)
    saveData(keyframe_w2kf_array, outdir=input_dir, fn='estimate_c.txt')

    # task (d)
    keyframe_w2kf_array = np.load('{}/keyframe_w2kf_array.npy'.format(output_dir))
    kfs, deltas, errors = taskD(K, input_dir, keyframes=keyframe_w2kf_array)
    if len(kfs) > 0:
        np.save('{}/keyframe_d'.format(output_dir), kfs)
        saveData(kfs, outdir=input_dir, fn='estimate_d.txt')

    # task (e)
    keyframe_e = taskE(K, input_dir, keyframe_w2kf_array, colors=rgbs, depths=depths, timestamp_color=t1, timestamp_depth=t2)
    if len(keyframe_e) > 0:
        np.save('{}/keyframe_e'.format(output_dir), keyframe_e)
        saveData(keyframe_e, outdir=input_dir, fn='estimate_e.txt')


def show(fname):
    im = imReadByGray(file_path=fname)
    plt.imshow(im, cmap='gray')
    # plt.show()


def tab():
    input_dir = 'data'
    output_dir = 'output'

    if os.path.isdir(input_dir):
        os.listdir(input_dir)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # load depth, rgb timestamp
    t1, rgbs, t2, depths = loadData(input_dir)
    logD('timestamp: {}, rgb: {}, depth: {}'.format(t1.shape, rgbs.shape, depths.shape))
    # actual parameter, which is copy from visiom.tum
    K = np.array([[520.9, 0, 325.1], [0, 521.0, 249.7], [0, 0, 1]])

    # task (a), (b)
    delta_x_array, pose_w2kf_array, distance_array = taskAB(K,
                                                            colors=rgbs,
                                                            depths=depths,
                                                            input_dir=input_dir,
                                                            timestamp_color=t1,
                                                            timestampe_depth=t2
                                                            , batch_size=len(rgbs)
                                                            )
    if len(pose_w2kf_array) > 0:
        np.save('{}/delta_xs_array'.format(output_dir), delta_x_array)
        np.save('{}/pose_w2kf_array'.format(output_dir), pose_w2kf_array)
        np.save('{}/distance_array'.format(output_dir), distance_array)
        saveData(pose_w2kf_array, outdir=input_dir, fn='estimate_ab_{}.txt'.format(.052))


def tc():
    input_dir = 'data'
    output_dir = 'output'

    if os.path.isdir(input_dir):
        os.listdir(input_dir)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # load depth, rgb timestamp
    t1, rgbs, t2, depths = loadData(input_dir)
    logD('timestamp: {}, rgb: {}, depth: {}'.format(t1.shape, rgbs.shape, depths.shape))
    # actual parameter, which is copy from visiom.tum
    K = np.array([[520.9, 0, 325.1], [0, 521.0, 249.7], [0, 0, 1]])

    # task (c)
    keyframe_w2kf_array, entropy_array, kf_idx_array = taskC(K,
                                                             input_dir=input_dir,
                                                             colors=rgbs,
                                                             depths=depths,
                                                             timestamp_color=t1,
                                                             timestampe_depth=t2,
                                                             lower=0.91,  # 0.915
                                                             upper=1.04
                                                             # , batch_size=len(rgbs)
                                                             )
    if len(keyframe_w2kf_array) > 0:
        np.save('{}/keyframe_w2kf_array'.format(output_dir), keyframe_w2kf_array)
        np.save('{}/entropy_array'.format(output_dir), entropy_array)
        np.save('{}/kf_idx_array'.format(output_dir), kf_idx_array)
        saveData(keyframe_w2kf_array, outdir=input_dir, fn='estimate_c.txt')


if __name__ == '__main__':
    main(sys.argv[1:])
    # plt.show()
    # import threading
    #
    # td1 = threading.Thread(target=tab)
    # td2 = threading.Thread(target=tc)
    # td1.start()
    # td2.start()
    print('finished....')
