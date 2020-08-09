import numpy as np
from scipy.spatial.transform import Rotation
from scipy.linalg import inv
from matplotlib import pyplot as plt

# self defined function
from utils.ImageUtils import imReadByGray
from utils.alignment import doAlignment
from utils.dataset import saveData
from utils.se3 import se3Exp
from utils.debug import logD
from utils.debug import logV
from utils.ImageUtils import downscale
from utils.calcResiduals import relativeError


def taskA(K, colors, depths, t1, input_dir='./data', output_dir='./output'):
    # save result for (a)
    i = np.random.randint(0, high=len(colors) - 1)
    c1 = np.double(imReadByGray('{}/{}'.format(input_dir, colors[i])))
    d1 = np.double(imReadByGray('{}/{}'.format(input_dir, depths[i]))) / 5000
    # draw figure
    fig = plt.figure(figsize=(12, 6))
    plt.title(t1[i])
    for i in np.arange(1, 5):
        si, sd, sk = downscale(c1, d1, K, i)
        plt.subplot(2, 4, i)
        plt.imshow(si)
        plt.title('scale level = {}'.format(i))
        plt.subplot(2, 4, i + 4)
        plt.imshow(sd, cmap='gray')
        plt.title('depth(scale={})'.format(i))
    fig.savefig('{}/downscale-a.png'.format(output_dir))
    logV('taskA is finished.....')


def taskB(K, colors, depths, t1, input_dir='./data', output_dir='./output', batch_size=500, threshold=0.052):
    timestamp = t1
    result_array, delta_x_array = [], []
    delta_xs_epoch, results_epoch = [], []
    trans_dist = []

    start, idx_kf, need_kf = 0, 0, 0
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

        # compute the reference frame with the keyframe
        c2 = np.double(imReadByGray('{}/{}'.format(input_dir, colors[i])))
        d2 = np.double(imReadByGray('{}/{}'.format(input_dir, depths[i]))) / 5000
        xis, errors, _ = doAlignment(ref_img=ckf, ref_depth=dkf, t_img=c2, t_depth=d2, k=K)
        xi = xis[-1]
        delta_xs_epoch.append(xi)
        delta_x_array.append(xi)
        logD('{} | {:04d} -> xi: {}'.format('taskB', i + 1, ['%-.08f' % x for x in xi]))

        # compute relative transform matrix
        t_inverse = inv(se3Exp(xi))  # just make sure current frame to keyframe
        distance = translation_distance(t_inverse)
        trans_dist.append(distance)
        need_kf = need_kf + 1
        # choose one frame from each N frames as keyframe
        if distance > threshold or need_kf >= 15:
            # here just choose the keyframe
            last_keyframe_pose = last_keyframe_pose @ t_inverse
            ckf, dkf = c2, d2
            idx_kf = i
            need_kf = 0

        current_frame_pose = last_keyframe_pose @ t_inverse
        R = current_frame_pose[:3, :3]  # rotation matrix
        t = current_frame_pose[:3, 3]  # t
        q = Rotation.from_matrix(R).as_quat()
        result = np.concatenate(([timestamp[i]], t, q))
        result = [eval('%.08f' % x) for x in result]
        results_epoch.append(result)
        result_array.append(result)
        logV('{} | pose({:04d} -> {:04d}) = {:.06f}\n\t{}'.format('taskB', i, idx_kf, distance, result))

    if len(result_array) > 0:
        np.save('{}/delta_xs_array'.format(output_dir), delta_x_array)
        np.save('{}/pose_w2kf_array'.format(output_dir), result_array)
        np.save('{}/distance_array'.format(output_dir), trans_dist)
        saveData(result_array, outdir=output_dir, fn='kf_estimate_b.txt')
        saveData(trans_dist, outdir=output_dir, fn='dist_estimate_b.txt')
    return delta_x_array, result_array, trans_dist


def taskC(K, colors, depths, t1, input_dir='./data', output_dir='./output', batch_size=500, lower=.9, upper=1.1):
    timestamp = t1
    keyframe_array, xi_array = [], []
    entropy_ratio, keyframe_idx_array = [], []

    start, need_kf = 0, 0
    for i in np.arange(start, batch_size):
        if i == 0:
            # initial result
            tmp = [timestamp[i], 0, 0, 0, 0, 0, 0, 1]
            keyframe_array.append(['%-.06f' % x for x in tmp])
            # world-frame initial pose
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
        need_kf = need_kf + 1
        if current_rate < lower or current_rate > upper or need_kf >= 15:
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
            need_kf = 0
            logV('{} | ({:04d} -> {}) result: {}'.format('taskC', i, key_frame_index, kf))

        entropy_ratio.append(current_rate)
        logV('{} | entropy of ({:04d} -> {:04d}) = {}'.format('taskC', i, key_frame_index, current_rate))

    if len(keyframe_array) > 0:
        np.save('{}/keyframe_w2kf_array'.format(output_dir), keyframe_array)
        np.save('{}/entropy_array'.format(output_dir), entropy_ratio)
        np.save('{}/kf_idx_array'.format(output_dir), keyframe_idx_array)
        saveData(keyframe_array, outdir=output_dir, fn='kf_estimate_c.txt')
        saveData(entropy_ratio, outdir=output_dir, fn='alpha_estimate_c.txt')
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


def taskE(K, input_dir, output_dir, kf, rgb, depth, t1):
    pass


def translation_distance(relative_pose):
    t = relative_pose[:3, 3]
    return np.sqrt((np.sum(t ** 2)))
