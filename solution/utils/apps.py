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


def method01(K, colors, depths, t1, input_dir='./data', output_dir='./output', batch_size=500):
    timestamp = t1
    pose_estimate, kf_estimate, delta_xs = [], [], []
    delta_xs_epoch, results_epoch = [], []
    distance_trans = []

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
        delta_xs.append(xi)
        logD('{} | {:04d} -> xi: {}'.format('method01', i + 1, ['%-.08f' % x for x in xi]))

        # compute relative transform matrix
        t_inverse = inv(se3Exp(xi))  # just make sure current frame to keyframe

        # choose one frame from each N frames as keyframe
        if i % 9 == 0:
            # here just choose the keyframe
            last_keyframe_pose = last_keyframe_pose @ t_inverse
            ckf, dkf = c2, d2
            idx_kf = i
            kf_R = last_keyframe_pose[:3, :3]  # rotation matrix
            kf_t = last_keyframe_pose[:3, 3]  # t
            kf_q = Rotation.from_matrix(kf_R).as_quat()
            tmp = np.concatenate(([timestamp[i]], kf_t, kf_q))
            tmp = [eval('%.08f' % x) for x in tmp]
            kf_estimate.append(tmp)

        current_frame_pose = last_keyframe_pose @ t_inverse
        R = current_frame_pose[:3, :3]  # rotation matrix
        t = current_frame_pose[:3, 3]  # t
        q = Rotation.from_matrix(R).as_quat()
        result = np.concatenate(([timestamp[i]], t, q))
        result = [eval('%.08f' % x) for x in result]
        results_epoch.append(result)
        pose_estimate.append(result)
        logV('{} | pose({:04d} -> {:04d}) =\n\t{}'.format('method-01', i, idx_kf, result))

    if len(pose_estimate) > 0:
        np.save('{}/delta_xs'.format(output_dir), delta_xs)
        np.save('{}/pose_estimate_1'.format(output_dir), pose_estimate)
        np.save('{}/kf_estimate_1'.format(output_dir), kf_estimate)
        np.save('{}/distance_trans'.format(output_dir), distance_trans)
        saveData(pose_estimate, outdir=output_dir, fn='pose_estimate_b0.txt')
        # saveData(trans_dist, outdir=output_dir, fn='dist_estimate_b.txt')
    return delta_xs, pose_estimate, distance_trans


def method02(K, colors, depths, t1, input_dir='./data', output_dir='./output', batch_size=500, d=0.052, a=0.012):
    timestamp = t1
    pose_estimate, kf_estimate, delta_xs = [], [], []
    delta_xs_epoch, results_epoch = [], []
    distance_trans = []

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
        delta_xs.append(xi)
        logD('{} | {:04d} -> xi: {}'.format('method-02', i + 1, ['%-.08f' % x for x in xi]))

        # compute relative transform matrix
        t_inverse = inv(se3Exp(xi))  # just make sure current frame to keyframe
        distance = translation_distance(t_inverse)
        distance_trans.append(distance)
        need_kf = need_kf + 1
        # choose one frame from each N frames as keyframe
        if distance > d or need_kf >= 15:
            # here just choose the keyframe
            last_keyframe_pose = last_keyframe_pose @ t_inverse
            ckf, dkf = c2, d2
            idx_kf = i
            need_kf = 0
            kf_R = last_keyframe_pose[:3, :3]  # rotation matrix
            kf_t = last_keyframe_pose[:3, 3]  # t
            kf_q = Rotation.from_matrix(kf_R).as_quat()
            tmp = np.concatenate(([timestamp[i]], kf_t, kf_q))
            tmp = [eval('%.08f' % x) for x in tmp]
            kf_estimate.append(tmp)

        current_frame_pose = last_keyframe_pose @ t_inverse
        R = current_frame_pose[:3, :3]  # rotation matrix
        t = current_frame_pose[:3, 3]  # t
        q = Rotation.from_matrix(R).as_quat()
        result = np.concatenate(([timestamp[i]], t, q))
        result = [eval('%.08f' % x) for x in result]
        results_epoch.append(result)
        pose_estimate.append(result)
        logV('{} | pose({:04d} -> {:04d}) = {:.06f}\n\t{}'.format('method-02', i, idx_kf, distance, result))

    if len(pose_estimate) > 0:
        np.save('{}/delta_xs_2'.format(output_dir), delta_xs)
        np.save('{}/pose_estimate_2'.format(output_dir), pose_estimate)
        np.save('{}/kf_estimate_2'.format(output_dir), kf_estimate)
        np.save('{}/distance_trans'.format(output_dir), distance_trans)
        saveData(pose_estimate, outdir=output_dir, fn='pose_estimate_2.txt')
        # saveData(trans_dist, outdir=output_dir, fn='dist_estimate_b.txt')
    return delta_xs, pose_estimate, distance_trans


def method03(K, colors, depths, t1, input_dir='./data', output_dir='./output', batch_size=500, threshold=.9):
    timestamp = t1
    kf_estimate, xi_array = [], []
    entropy_ratio, kf_idx = [], []

    start, need_kf = 0, 0
    for i in np.arange(start, batch_size):
        if i == 0:
            # initial result
            tmp = [timestamp[i], 0, 0, 0, 0, 0, 0, 1]
            kf_estimate.append(['%-.06f' % x for x in tmp])
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
        if current_rate < threshold or need_kf >= 15:
            # here just choose the keyframe & update keyframe
            last_keyframe_pose = last_keyframe_pose @ t_inverse
            ckf, dkf = c2, d2
            key_frame_index = i
            kf_idx.append(i)
            logD('keyframe_index: {}\n\tctx: {}'.format(i, last_keyframe_pose))
            base_line = H_xi
            current_rate = H_xi / base_line
            # change the pose of last keyframe to new format, and add to list
            R = last_keyframe_pose[:3, :3]  # rotation matrix
            t = last_keyframe_pose[:3, 3]  # t
            q = Rotation.from_matrix(R).as_quat()
            tmp = np.concatenate(([timestamp[i]], t, q))
            kf = [eval('%-.08f' % x) for x in tmp]
            kf_estimate.append(kf)
            need_kf = 0
            logV('{} | ({:04d} -> {}) result: {}'.format('method-03', i, key_frame_index, kf))

        entropy_ratio.append(current_rate)
        logV('{} | entropy of ({:04d} -> {:04d}) = {}'.format('method-03', i, key_frame_index, current_rate))

    if len(kf_estimate) > 0:
        np.save('{}/kf_estimate_3'.format(output_dir), kf_estimate)
        np.save('{}/entropy_rate'.format(output_dir), entropy_ratio)
        np.save('{}/kf_idx'.format(output_dir), kf_idx)
        saveData(kf_estimate, outdir=output_dir, fn='kf_estimate_3.txt')
        # saveData(entropy_ratio, outdir=output_dir, fn='alpha_estimate_c.txt')
    return kf_estimate, entropy_ratio, kf_idx


def taskD(K, input_dir, keyframes, kf_idx, rgbs, depths, t1):
    kfs, deltas, errors = [], [], []
    # (d) optimization of keyframe pose
    for i, kf_i, idx in enumerate(zip(keyframes, kf_idx)):
        d, e = [], []
        for j in [i - 1, i, i + 1]:
            if j < 0 or j > len(keyframes) - 1:
                j = (j + len(keyframes)) % len(keyframes)
            kf_j = keyframes[j]
            xis, errors, _ = doAlignment(ref_img=rgbs[idx], ref_depth=depths[idx], t_img=rgbs[kf_idx[j]], t_depth=depths[kf_idx[j]], k=K)
            kf_pose = np.identity(4)
            t_inverse = inv(se3Exp(xis[-1]))
            kf = kf_pose @ t_inverse
            T1, T2, _, error = relativeError(kf_ref=kf_i, kf=kf_j, delta=kf)
            # d.append(delta)
            e.append(error)
        # d = np.asarray(d).mean(axis=0)
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
