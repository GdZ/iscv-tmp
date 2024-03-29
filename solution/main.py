#! -*- coding: utf-8 -*-
# !/bin/env python
import os
import sys
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as Rfunc
from scipy.linalg import inv

# self defined function
from utils.dataset import loadData
from utils.ImageUtils import imReadByGray
from utils.alignment import doAlignment
from utils.se3 import se3Exp
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
    """
    :param input_dir:
    :param t1: timestamp of rgb image
    :param rgbs: rgb-images
    :param t2: timestamp of depth images
    :param depths: depth-images
    """
    results, xi_arr = [], []
    start, step = 0,9
    threshold = 0.9
    # actual parameter, which is copy from visiom.tum
    K = np.array([[520.9, 0, 325.1], [0, 521.0, 249.7], [0, 0, 1]])
    entropy_ratio = []
    key_frame_indices= []
    # for i in np.arange(start, len(rgbs)):
    for i in np.arange(start, 100):
        if i == 0:
            # write the head of the estimate.txt
            with open('data/estimate.txt', "w") as f:
                f.write('# timestamp tx ty tz qx qy qz qw\n')
            f.close()
            with open('data/delta_x.csv', 'w') as f:
                f.write('')
            f.close()
            #
            tmp = [t1[i], 0, 0, 0, 0, 0, 0, 1]
            results.append(['%-.06f' % x for x in tmp])
            # world-frame initial pose
            pw = np.array([0, 0, 0, 1])
            last_keyframe_pose = np.identity(4)
            c1 = np.double(imReadByGray('{}/{}'.format(input_dir, rgbs[i])))
            d1 = np.double(imReadByGray('{}/{}'.format(input_dir, depths[i]))) / 5000
            ckf, dkf = c1, d1
            key_frame_index = 0

        # compute the reference frame with the keyframe
        c2 = np.double(imReadByGray('{}/{}'.format(input_dir, rgbs[i])))
        d2 = np.double(imReadByGray('{}/{}'.format(input_dir, depths[i]))) / 5000
        xis, errors, H_xi = doAlignment(ref_img=ckf, ref_depth=dkf, t_img=c2, t_depth=d2, k=K)
        xi = xis[-1]
        xi_arr.append(xi)
        logV('{:04d} -> xi: {}'.format(i+1, ['%-.08f' % x for x in xi]))

        if i == 0:
            base_line = H_xi
            plt.imshow(ckf,cmap='gray')
            plt.show()

        # compute relative transform matrix
        Tinv = inv(se3Exp(xi))   # just make sure current frame to keyframe
        #if i % step == 0:
        if i == key_frame_index + 1:
            base_line = H_xi
        if H_xi/base_line < threshold:
            # here just choose the keyframe & update keyframe
            last_keyframe_pose = last_keyframe_pose @ Tinv
            ckf, dkf = c2, d2
            key_frame_index = i
            key_frame_indices.append(i)
            print(i)
            base_line = H_xi
            plt.imshow(c2, cmap='gray')
            plt.show()

        current_frame_pose = last_keyframe_pose @ Tinv
        R = current_frame_pose[:3, :3]  # rotation matrix
        t = current_frame_pose[:3, 3]  # t
        q = Rfunc.from_matrix(R).as_quat()
        result = np.concatenate(([t1[i]], t, q))
        results.append(['%-.08f' % x for x in result])

        # entropy ratio
        entropy_ratio.append(H_xi/base_line)
        logV('{:04d} -> resutl: {}'.format(i+1, ['%-.08f' % x for x in result]))
        print(key_frame_index)

        # save result to 'data/estimate.txt'
        if i % step == 0:
            # save \delta_{x}
            delta_x = pd.DataFrame(np.asarray(xi_arr))
            delta_x.to_csv('data/delta_x.csv', encoding='utf-8', index_label=False, index=False, mode='a', header=False)
            # save estimate.txt
            csv = pd.DataFrame(np.asarray(results), columns=['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
            csv.to_csv('data/estimate.txt', encoding='utf-8', index_label=False, index=False, sep=' ', mode='a', header=False)
            results, xi_arr = [], []
    # plot entropy
    plt.plot(entropy_ratio, 'r-')

def show(fname):
    im = imReadByGray(file_path=fname)
    plt.imshow(im, cmap='gray')
    # plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
    plt.show()
    print('finished....')
