import numpy as np
from scipy.spatial.transform import Rotation
from scipy.linalg import inv
from matplotlib import pyplot as plt

# self defined function
from utils.ImageUtils import imReadByGray
from utils.alignment import doAlignment
from utils.se3 import se3Exp
from utils.debug import logD
from utils.debug import logV
from utils.ImageUtils import downscale
from utils.calcResiduals import relativeError


def taskAB(K, colors, depths, timestamp_color, timestampe_depth,
           input_dir='./data', epoch_size=9, batch_size=500, threshold=0.052):
    timestamp = timestamp_color
    result_array, delta_x_array = [], []
    delta_xs_epoch, results_epoch = [], []
    trans_dist = []

    start, idx_kf = 0, 0
    for i in np.arange(start, batch_size):
        if i == 0:
            # initial result
            tmp = [timestamp[i], 0, 0, 0, 0, 0, 0, 1]
            results_epoch.append(['%-.06f' % x for x in tmp])
            # world-frame initial pose
            pw = np.array([0, 0, 0, 1])
            last_keyframe_pose = np.identity(4)
            c1 = np.double(imReadByGray('{}/{}'.format(input_dir, colors[i])))
            d1 = np.double(imReadByGray('{}/{}'.format(input_dir, depths[i]))) / 5000
            ckf, dkf = c1, d1
            # save result for (a)
            # fig = plt.figure(figsize=(12, 6))
            # for i in np.arange(1, 5):
            #     si, sd, sk = downscale(c1, d1, K, i)
            #     plt.subplot(2,4,i)
            #     plt.imshow(si)
            #     plt.title('scale level = {}'.format(i))
            #     plt.subplot(2,4,i+4)
            #     plt.imshow(sd, cmap='gray')
            #     plt.title('depth(scale={})'.format(i))

        # compute the reference frame with the keyframe
        c2 = np.double(imReadByGray('{}/{}'.format(input_dir, colors[i])))
        d2 = np.double(imReadByGray('{}/{}'.format(input_dir, depths[i]))) / 5000
        xis, errors, _ = doAlignment(ref_img=ckf, ref_depth=dkf, t_img=c2, t_depth=d2, k=K)
        xi = xis[-1]
        delta_xs_epoch.append(xi)
        delta_x_array.append(xi)
        # logV('{:04d} -> xi: {}'.format(i + 1, ['%-.08f' % x for x in xi]))

        # compute relative transform matrix
        t_inverse = inv(se3Exp(xi))  # just make sure current frame to keyframe
        distance = translation_distance(t_inverse)
        trans_dist.append(distance)
        # choose one frame from each N frames as keyframe
        if distance > threshold:
            # here just choose the keyframe
            last_keyframe_pose = last_keyframe_pose @ t_inverse
            ckf, dkf = c2, d2
            idx_kf = i

        current_frame_pose = last_keyframe_pose @ t_inverse
        R = current_frame_pose[:3, :3]  # rotation matrix
        t = current_frame_pose[:3, 3]  # t
        q = Rotation.from_matrix(R).as_quat()
        result = np.concatenate(([timestamp[i]], t, q))
        results_epoch.append([eval('%-.08f' % x) for x in result])
        result_array.append([eval('%-.08f' % x) for x in result])

        logV('pose({:04d} -> {:04d}) = {:.06f}\n\t{}'.format(i, idx_kf, distance, ['%-.08f' % x for x in result]))

    return delta_x_array, result_array, trans_dist


def taskC(K, colors, depths, timestamp_color, timestampe_depth,
          input_dir='./data', epoch_size=9, batch_size=500, lower=.9, upper=1.1):
    timestamp = timestamp_color
    keyframe_array, xi_array = [], []
    entropy_ratio, keyframe_idx_array = [], []

    start = 0
    for i in np.arange(start, batch_size):
        if i == 0:
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
        logD('{:04d} -> xi: {}'.format(i, ['%-.08f' % x for x in xi]))

        if i == 0:
            base_line = H_xi
            # plt.imshow(ckf, cmap='gray')
            # plt.show()

        # compute relative transform matrix
        t_inverse = inv(se3Exp(xi))  # just make sure current frame to keyframe

        if i == key_frame_index + 1:
            base_line = H_xi

        # entropy ratio, save entropy of all images
        current_rate = H_xi / base_line
        if current_rate < lower or current_rate > upper:
            # here just choose the keyframe & update keyframe
            last_keyframe_pose = last_keyframe_pose @ t_inverse
            ckf, dkf = c2, d2
            key_frame_index = i
            keyframe_idx_array.append(i)
            logD('keyframe_index: {}\n\tctx: {}'.format(i, last_keyframe_pose))
            base_line = H_xi
            current_rate = H_xi / base_line
            # change the pose of last keyframe to new format, and add to list
            R = last_keyframe_pose[:3, :3]  # rotation matrix
            t = last_keyframe_pose[:3, 3]  # t
            q = Rotation.from_matrix(R).as_quat()
            tmp = np.concatenate(([timestamp[i]], t, q))
            kf = [eval('%-.08f' % x) for x in tmp]
            keyframe_array.append(kf)
            logV('{:04d} -> idx_kf: {} result: {}'.format(i, key_frame_index, kf))

        entropy_ratio.append(current_rate)
        logV('entropy of ({:04d} -> {:04d}) = {}'.format(i, key_frame_index, current_rate))

    return keyframe_array, entropy_ratio, keyframe_idx_array


def taskD(K, input_dir, keyframes):
    kfs, deltas, errors = [], [], []
    # (d) optimization of keyframe pose
    for i, kf_i in enumerate(keyframes):
        d, e = [], []
        for j in [i - 1, i, i + 1]:
            if j < 0 or j > len(keyframes) - 1:
                j = (j + len(keyframes)) % len(keyframes)
            kf_j = keyframes[j]
            T1, T2, delta, error = relativeError(trans_kf1=kf_i, trans_kf2=kf_j)
            d.append(delta)
            e.append(error)

        d = np.asarray(d).mean(axis=0)
        e = np.asarray(e).mean(axis=0)
        t_inverse = inv(se3Exp(e))
        T1 = T1 @ t_inverse
        R, t = T1[:3, :3], T1[:3, 3]
        q = Rotation.from_matrix(R).as_quat()
        kf = np.concatenate(([kf_i[0]], t, q))
        kf = [eval('{:08f}'.format(x)) for x in kf]
        kfs.append(kf)
        deltas.append(d)
        errors.append(e)

    return kfs, deltas, errors


def taskE(K, input_dir, keyframes_color, keyframes_depth, timestamp_color):
    pass


def translation_distance(relative_pose):
    t = relative_pose[:3, 3]
    return np.sqrt((np.sum(t ** 2)))
