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
from utils.ImageUtils import alignment


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


def show(fname):
    im = imreadbw(fname=fname)
    plt.imshow(im, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
