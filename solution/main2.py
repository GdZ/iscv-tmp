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
    # actual parameter, which is copy from visiom.tum
    K = np.array([[520.9, 0, 325.1], [0, 521.0, 249.7], [0, 0, 1]])

    # task (a), (b)
    delta_x_array, pose_w_kf_array = taskAB(K, input_dir,
                                            colors=rgbs, depths=depths, timestamp_color=t1, timestampe_depth=t2,
                                            epoch_size=9, batch_size=100)
    np.save('delta_xs_array', delta_x_array)
    np.save('pose_w2kf_array', pose_w_kf_array)

    # task (c)
    keyframe_w_kf_array, entropy_array, kf_idx_array = taskC(K, input_dir,
                                                     colors=rgbs, depths=depths,
                                                     timestamp_color=t1, timestampe_depth=t2,
                                                     threshold=0.9,
                                                     epoch_size=9, batch_size=100)
    np.save('keyframe_w2kf_array', keyframe_w_kf_array)
    np.save('entropy_array', entropy_array)
    np.save('kf_idx_array', kf_idx_array)

    # task (d)
    # keyframe_d = taskD(K, input_dir, kf_idx_c, colors=rgbs, depths=depths, timestamp_color=t1, timestamp_depth=t2)

    # task (e)
    # keyframe_e = taskE(K, input_dir, kf_idx_c, colors=rgbs, depths=depths, timestamp_color=t1, timestamp_depth=t2)


def taskAB(K, input_dir, colors, depths, timestamp_color, timestampe_depth, epoch_size=9, batch_size=100):
    """
    :param K:
    :param input_dir:
    :param colors:
    :param depths:
    :param timestamp_color:
    :param timestampe_depth:
    """
    timestamp = timestamp_color
    result_array, delta_x_array = [], []
    delta_xs_epoch, results_epoch = [], []

    start = 0
    for i in np.arange(start, batch_size):
        if i == 0:
            # write the head of the estimate.txt
            with open('{}/estimate.txt'.format(input_dir), "w") as f:
                f.write('# timestamp tx ty tz qx qy qz qw\n')
            f.close()
            with open('{}/delta_x.csv'.format(input_dir), 'w') as f:
                f.write('')
            f.close()
            # initial result
            tmp = [timestamp[i], 0, 0, 0, 0, 0, 0, 1]
            results_epoch.append(['%-.06f' % x for x in tmp])
            # world-frame initial pose
            pw = np.array([0, 0, 0, 1])
            last_keyframe_pose = np.identity(4)
            c1 = np.double(imReadByGray('{}/{}'.format(input_dir, colors[i])))
            d1 = np.double(imReadByGray('{}/{}'.format(input_dir, depths[i]))) / 5000
            ckf, dkf = c1, d1

        # compute the reference frame with the keyframe
        c2 = np.double(imReadByGray('{}/{}'.format(input_dir, colors[i])))
        d2 = np.double(imReadByGray('{}/{}'.format(input_dir, depths[i]))) / 5000
        xis, errors, _ = doAlignment(ref_img=ckf, ref_depth=dkf, t_img=c2, t_depth=d2, k=K)
        xi = xis[-1]
        delta_xs_epoch.append(xi)
        delta_x_array.append(xi)
        logV('{:04d} -> xi: {}'.format(i + 1, ['%-.08f' % x for x in xi]))

        # compute relative transform matrix
        t_inverse = inv(se3Exp(xi))  # just make sure current frame to keyframe

        # choose one frame from each N frames as keyframe
        if i % epoch_size == 0:
            # here just choose the keyframe
            last_keyframe_pose = last_keyframe_pose @ t_inverse
            ckf, dkf = c2, d2

        current_frame_pose = last_keyframe_pose @ t_inverse
        R = current_frame_pose[:3, :3]  # rotation matrix
        t = current_frame_pose[:3, 3]  # t
        q = Rfunc.from_matrix(R).as_quat()
        result = np.concatenate(([timestamp[i]], t, q))
        results_epoch.append(['%-.08f' % x for x in result])
        result_array.append(['%-.08f' % x for x in result])
        logV('{:04d} -> resutl: {}'.format(i + 1, ['%-.08f' % x for x in result]))

        # save result to 'data/estimate.txt'
        if i % epoch_size == 0:
            # save \delta_{x}
            delta_x = pd.DataFrame(np.asarray(delta_xs_epoch))
            delta_x.to_csv('{}/delta_x.csv'.format(input_dir), encoding='utf-8', index_label=False, index=False,
                           mode='a', header=False)
            # save estimate.txt
            csv = pd.DataFrame(np.asarray(results_epoch),
                               columns=['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
            csv.to_csv('{}/estimate.txt'.format(input_dir), encoding='utf-8', index_label=False, index=False, sep=' ',
                       mode='a', header=False)
            results_epoch, delta_xs_epoch = [], []

    return delta_x_array, result_array


def taskC(K, input_dir, colors, depths, timestamp_color, timestampe_depth, epoch_size=9, batch_size=100, threshold=.9):
    """
    :param K:
    :param input_dir:
    :param colors:
    :param depths:
    :param timestamp_color:
    :param timestampe_depth:
    :return:
    :rtype: object
    """
    timestamp = timestamp_color
    keyframe_array, xi_array = [], []
    entropy_ratio, keyframe_idx_array = [], []

    start = 0
    for i in np.arange(start, batch_size):
        if i == 0:
            # write the head of the estimate.txt
            with open('{}/estimate.txt'.format(input_dir), "w") as f:
                f.write('# timestamp tx ty tz qx qy qz qw\n')
            f.close()
            with open('{}/delta_x.csv'.format(input_dir), 'w') as f:
                f.write('')
            f.close()
            # initial result
            tmp = [timestamp[i], 0, 0, 0, 0, 0, 0, 1]
            keyframe_array.append(['%-.06f' % x for x in tmp])
            # world-frame initial pose
            pw = np.array([0, 0, 0, 1])
            last_keyframe_pose = np.identity(4)
            c1 = np.double(imReadByGray('{}/{}'.format(input_dir, colors[i])))
            d1 = np.double(imReadByGray('{}/{}'.format(input_dir, depths[i]))) / 5000
            ckf, dkf = c1, d1
            key_frame_index = 0

        # compute the reference frame with the keyframe
        c2 = np.double(imReadByGray('{}/{}'.format(input_dir, colors[i])))
        d2 = np.double(imReadByGray('{}/{}'.format(input_dir, depths[i]))) / 5000
        xis, errors, H_xi = doAlignment(ref_img=ckf, ref_depth=dkf, t_img=c2, t_depth=d2, k=K)
        xi = xis[-1]
        xi_array.append(xi)
        logD('{:04d} -> xi: {}'.format(i + 1, ['%-.08f' % x for x in xi]))

        if i == 0:
            base_line = H_xi
            # plt.imshow(ckf, cmap='gray')
            # plt.show()

        # compute relative transform matrix
        t_inverse = inv(se3Exp(xi))  # just make sure current frame to keyframe

        if i == key_frame_index + 1:
            base_line = H_xi

        # entropy ratio, save entropy of all images
        entropy_ratio.append(H_xi / base_line)
        logV('entropy of ({:04d} -> {:04d}) = {}'.format(i + 1, key_frame_index, H_xi / base_line))

        if (H_xi / base_line) < threshold:
            # here just choose the keyframe & update keyframe
            last_keyframe_pose = last_keyframe_pose @ t_inverse
            ckf, dkf = c2, d2
            key_frame_index = i
            keyframe_idx_array.append(i)
            logD('keyframe_index: {}\n\tctx: {}'.format(i, last_keyframe_pose))
            base_line = H_xi
            # change the pose of last keyframe to new format, and add to list
            R = last_keyframe_pose[:3, :3]  # rotation matrix
            t = last_keyframe_pose[:3, 3]  # t
            q = Rfunc.from_matrix(R).as_quat()
            result = np.concatenate(([timestamp[i]], t, q))
            keyframe_array.append(['%-.08f' % x for x in result])
            # logV('{:04d} -> result: {}'.format(i + 1, ['%-.08f' % x for x in result]))
            logV('{:04d} -> idx_kf: {} result: {}'.format(i + 1, key_frame_index, ['%-.08f' % x for x in result]))

    return keyframe_array, entropy_ratio, keyframe_idx_array


def taskD(K, input_dir, keyframes_color, keyframes_depth, timestamp_color):
    """
    :param K:
    :param input_dir:
    :param keyframes_color:
    :param keyframes_depth:
    :param timestamp_color:
    :return:
    :rtype: object
    """
    xi_array, result_array = [], []
    # (d) optimization of keyframe pose
    for i in np.arange(len(keyframes_color) - 1):
        last_ref_ckf, last_ref_dkf = keyframes_color(i), keyframes_depth(i)
        timg, tdep = keyframes_color(i + 1), keyframes_depth(i + 1)
        xis, errors, h_xi = doAlignment(ref_img=last_ref_ckf, ref_depth=last_ref_dkf, t_img=timg, t_depth=tdep, k=K)
        xi = xis[-1]
        logV('{:04d} -> xi: {}'.format(i + 1, ['%-.08f' % x for x in xi]))
        # new pose of keyframe
        # pass
    # recompute pose of all image
    return xi_array, result_array


def taskE(K, input_dir, keyframes_color, keyframes_depth, timestamp_color):
    pass


def show(fname):
    im = imReadByGray(file_path=fname)
    plt.imshow(im, cmap='gray')
    # plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
    plt.show()
    print('finished....')
