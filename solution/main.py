#! -*- coding: utf-8 -*-
# !/bin/env python
import os
import sys
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# self defined function
from utils.dataset import loadData
from utils.ImageUtils import imReadByGray
from utils.alignment import doAlignment
from utils.debug import isDebug
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
    timestamp_rgb, rgb, timestamp_depth, depth = loadData(input_dir)
    logD('timestamp: {}, rgb: {}, depth: {}'.format(timestamp_rgb.shape, rgb.shape, depth.shape))
    alignment(input_dir, t1=timestamp_rgb, rgbs=rgb, t2=timestamp_depth, depths=depth)


def alignment(input_dir, t1, rgbs, t2, depths):
    buffer = []


    start, step = 0, 9
    for i in np.arange(start, len(rgbs), step):
        results = []
        if i == 0:
            # write the head of the estimate.txt
            with open('data/estimate.txt', "w") as f:
                f.write('# timestamp tx ty tz qx qy qz qw\n')
            f.close()
            tmp = [t1[i], 0, 0, 0, 0, 0, 0, 1]
            results.append(['%-.06f' % x for x in tmp])

        if isDebug():
            # parameter just for testing, which is copy from matlab
            K = np.array([[517.3, 0, 318.6], [0, 516.5, 255.3], [0, 0, 1]])
            c1 = np.double(imReadByGray('{}/{}'.format(input_dir, 'rgb/1311868164.399026.png')))
            d1 = np.double(imReadByGray('{}/{}'.format(input_dir, 'depth/1311868164.407784.png'))) / 5000
            c2 = np.double(imReadByGray('{}/{}'.format(input_dir, 'rgb/1311868164.363181.png')))
            d2 = np.double(imReadByGray('{}/{}'.format(input_dir, 'depth/1311868164.373557.png'))) / 5000
            logD('c1.shape = ({}), d1.shape = ({})'.format(c1.shape, d1.shape))

            # % result:
            # % approximately  -0.0018    0.0065    0.0369   -0.0287   -0.0184   -0.0004
            logD('approximately  -0.0018    0.0065    0.0369   -0.0287   -0.0184   -0.0004')
            xis, errors = doAlignment(ref_img=c1, ref_depth=d1, t_img=c2, t_depth=d2, k=K)
            logD('timestamp: {}, error: {}, xi: {}'.format(t1[i], errors[-1], xis[-1]))
            result = np.zeros(8)
            result[0] = 1311868164.399026
            result[1:7] = xis[-1]
            results.append(['%-.06f' % x for x in result])
            break

        else:
            # actual parameter, which is copy from visiom.tum
            K = np.array([[520.9, 0, 325.1], [0, 521.0, 249.7], [0, 0, 1]])
            c1 = np.double(imReadByGray('{}/{}'.format(input_dir, rgbs[i])))
            d1 = np.double(imReadByGray('{}/{}'.format(input_dir, depths[i]))) / 5000
            logD('c1.shape = ({}), d1.shape = ({})'.format(c1.shape, d1.shape))

            # each 'step'-frame image depend on the 0-frame
            for j in np.arange(i+1, i+step):
                c2 = np.double(imReadByGray('{}/{}'.format(input_dir, rgbs[i + j])))
                d2 = np.double(imReadByGray('{}/{}'.format(input_dir, depths[i + j]))) / 5000
                xis, errors = doAlignment(ref_img=c1, ref_depth=d1, t_img=c2, t_depth=d2, k=K)
                logV('{:04d} timestamp: {:.07f}, error: {:.08f}, xi: {}'.format(j, t1[j], errors[-1], xis[-1]))
                result = np.zeros(8)
                result[0] = t1[j]
                result[1:7] = xis[-1]
                results.append(['%-.06f' % x for x in result])

        # save result to 'data/estimate.txt'
        csv = pd.DataFrame(np.asarray(results), columns=['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
        csv.to_csv('data/estimate.txt', encoding='utf-8', index_label=False, index=False, sep=' ', mode='a', header=False)
        # just compute first group
        # break


def show(fname):
    im = imReadByGray(file_path=fname)
    plt.imshow(im, cmap='gray')
    # plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
    plt.show()
    print('finished....')
