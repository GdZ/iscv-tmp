#! -*- coding: utf-8 -*-
# !/bin/env python
import os
import sys
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# self defined function
from utils.dataset import load_data
from utils.ImageUtils import imreadbw
from utils.alignment import do_alignment
from utils.debug import is_debug
from utils.debug import logD
from utils.debug import logV


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
    timestamp, rgb, depth = load_data(input_dir)
    logD('timestamp: {}, rgb: {}, depth: {}'.format(timestamp.shape, rgb.shape, depth.shape))
    alignment(input_dir, timestamp, rgbs=rgb, depths=depth)


def alignment(input_dir, timestamps, rgbs, depths):
    step = 9
    results = []

    for i in np.arange(0, len(rgbs), step):
        if is_debug():
            # parameter just for testing, which is copy from matlab
            K = np.array([[517.3, 0, 318.6], [0, 516.5, 255.3], [0, 0, 1]])
            c1 = np.double(imreadbw('{}/{}'.format(input_dir, 'rgb/1311868164.399026.png')))
            d1 = np.double(imreadbw('{}/{}'.format(input_dir, 'depth/1311868164.407784.png'))) / 5000
            c2 = np.double(imreadbw('{}/{}'.format(input_dir, 'rgb/1311868164.363181.png')))
            d2 = np.double(imreadbw('{}/{}'.format(input_dir, 'depth/1311868164.373557.png'))) / 5000
            # % result:
            # % approximately  -0.0018    0.0065    0.0369   -0.0287   -0.0184   -0.0004
            logD('approximately  -0.0018    0.0065    0.0369   -0.0287   -0.0184   -0.0004')
            xis, errors = do_alignment(c1, d1, c2, d2, K)
            logD('timestamp: {}, error: {}, xi: {}'.format(timestamps[i], errors[-1], xis[-1]))

        else:
            # actual parameter, which is copy from visiom.tum
            K = np.array([[520.9, 0, 325.1], [0, 521.0, 249.7], [0, 0, 1]])
            c1 = np.double(imreadbw('{}/{}'.format(input_dir, rgbs[i])))
            d1 = np.double(imreadbw('{}/{}'.format(input_dir, depths[i]))) / 5000

            # each 'step'-frame image depend on the 0-frame
            for j in np.arange(1, step):
                c2 = np.double(imreadbw('{}/{}'.format(input_dir, rgbs[i + j])))
                d2 = np.double(imreadbw('{}/{}'.format(input_dir, depths[i + j]))) / 5000
                xis, errors = do_alignment(c1, d1, c2, d2, K)
                logV('timestamp: {:.07f}, error: {:.08f}, xi: {}'.format(timestamps[j], errors[-1], xis[-1]))

        # just compute first group
        break


def show(fname):
    im = imreadbw(fname=fname)
    plt.imshow(im, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
