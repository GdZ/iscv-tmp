#! -*- coding: utf-8 -*-
# !/bin/env python
import time
import os
import sys
from shutil import copy
import argparse
import threading
import numpy as np
from matplotlib import pyplot as plt
# self defined function
from utils.dataset import loadData
from utils.ImageUtils import imReadByGray
from utils.dataset import saveData
from utils.debug import logD
from utils.debug import logV
from utils.apps import method02, method01, method03, taskD, taskE


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
    # alignment(input_dir, output_dir, t1=timestamp_rgb, rgbs=rgb, t2=timestamp_depth, depths=depth)
    alignmentsMulti(input_dir, output_dir, t1=timestamp_rgb, rgbs=rgb, t2=timestamp_depth, depths=depth)


def alignmentsMulti(input_dir, output_dir, t1, rgbs, t2, depths):
    K = np.array([[520.9, 0, 325.1], [0, 521.0, 249.7], [0, 0, 1]])
    output_dir = '{}/{}'.format(output_dir, time.time())
    os.mkdir(output_dir)
    prepare(output_dir)

    # parallel running task
    # (b)
    batch_size = 500
    # batch_size = len(rgbs)
    # for i, lvl in enumerate(np.arange(5, 0, -2)):
    # lvl = 5
    # td0 = threading.Thread(target=method01, args=(K, rgbs, depths, t1, input_dir, output_dir, batch_size, lvl))
    # td0.start()
    # td1 = threading.Thread(target=method02, args=(K, rgbs, depths, t1, input_dir, output_dir, batch_size, 0.12, 0.12, lvl))
    # td1.start()
    # td2 = threading.Thread(target=method03, args=(K, rgbs, depths, t1, input_dir, output_dir, batch_size, .9, lvl))
    # td2.start()
    # (d), (e)
    alignment(input_dir, output_dir, t1=t1, rgbs=rgbs, t2=t2, depths=depths)


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

    output_dir = 'output/1597067216.0324774'
    output_dir = 'output/1597495410.0373957'
    # output_dir = 'output/tmp'
    kf_idx = np.load('{}/kf_idx-5.npy'.format(output_dir)).astype(int)
    tmp = np.zeros(kf_idx.size + 1)
    tmp[1:] = kf_idx
    kf_idx = tmp
    kf_estimate_3 = np.load('{}/kf_estimate_3-5.npy'.format(output_dir)).astype(np.float)

    # task (d)
    kfs = taskD(K, keyframes=kf_estimate_3, kf_idx=kf_idx, rgbs=rgbs, depths=depths, t1=t1, input_dir=input_dir, output_dir=output_dir)
    # if len(kfs) > 0:
    #     np.save('{}/keyframe_d'.format(output_dir), kfs)
    #     saveData(kfs, outdir=input_dir, fn='estimate_d.txt')

    # task (e)
    # keyframe_e = taskE(K, keyframes=kf_estimate_3, kf_idx=kf_idx, colors=rgbs, depths=depths, t1=t1, input_dir=input_dir, output_dir=output_dir)
    # if len(keyframe_e) > 0:
    #     np.save('{}/keyframe_e'.format(output_dir), keyframe_e)
    #     saveData(keyframe_e, outdir=input_dir, fn='estimate_e.txt')


def show(fname):
    im = imReadByGray(file_path=fname)
    plt.imshow(im, cmap='gray')
    # plt.show()


def prepare(output_dir):
    # logV(os.listdir())
    # logV(os.listdir(output_dir))
    copy('{}/../makefile'.format(output_dir), output_dir)
    copy('{}/../associate.py'.format(output_dir), output_dir)
    copy('{}/../evaluate_ate_v1.py'.format(output_dir), output_dir)
    copy('{}/../evaluate_ate_v2.py'.format(output_dir), output_dir)
    copy('{}/../evaluate_rpe.py'.format(output_dir), output_dir)
    copy('{}/../groundtruth.txt'.format(output_dir), output_dir)


if __name__ == '__main__':
    main(sys.argv[1:])
    logV('finished....')
