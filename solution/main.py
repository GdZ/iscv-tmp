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
    print(timestamp.shape, rgb.shape, depth.shape)
    fname = '{}/{}'.format(input_dir, rgb[0])
    print(fname)
    # show(fname)
    alignment(input_dir, timestamp, rgbs=rgb, depths=depth)


def alignment(input_dir, timestamps, rgbs, depths):
    # Reference from website of vision tum
    K = np.array([[520.9, 0, 325.1], [0, 521.0, 249.7], [0, 0, 1]])

    step = 9
    results = []
    for i in np.arange(1, len(rgbs), step):
        c1 = np.double(imreadbw('{}/{}'.format(input_dir, rgbs[1])))
        d1 = np.double(imreadbw('{}/{}'.format(input_dir, depths[2]))) / 5000
        # c1 = np.double(imreadbw('{}/{}'.format(input_dir, rgbs[i])))
        # d1 = np.double(imreadbw('{}/{}'.format(input_dir, depths[i]))) / 5000
        for j in np.arange(1, step):
            c2 = np.double(imreadbw('{}/{}'.format(input_dir, rgbs[0])))
            d2 = np.double(imreadbw('{}/{}'.format(input_dir, depths[1]))) / 5000
            # c2 = np.double(imreadbw('{}/{}'.format(input_dir, rgbs[i + j])))
            # d2 = np.double(imreadbw('{}/{}'.format(input_dir, depths[i + j]))) / 5000
            # % result:
            # % approximately  -0.0018    0.0065    0.0369   -0.0287   -0.0184   -0.0004
            results.append({'timestamp': timestamps[i], 'result': do_alignment(c1, d1, c2, d2, K)})
            break
        break
    results = np.asarray(results)


def show(fname):
    im = imreadbw(fname=fname)
    plt.imshow(im, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
