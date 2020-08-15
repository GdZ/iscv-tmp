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
from utils.deriveResiduals import derivePoseGraphResidualsNumeric
from utils.alignment import poseGraph


def taskA(K, colors, depths, t1, input_dir='./data', output_dir='./output'):
    # save result for (a)
    # draw figure
    # fig = plt.figure(figsize=(12, 6))
    # plt.title(t1[i])
    datasets = []
    for j, t1, im, dep in enumerate(zip(t1, colors, depths)):
        c1 = np.double(imReadByGray('{}/{}'.format(input_dir, colors[j])))
        d1 = np.double(imReadByGray('{}/{}'.format(input_dir, depths[j]))) / 5000
        row = {'t1': t1, 'ds': []}
        for i, key in enumerate(['c5', 'c4', 'c3', 'c2']):
            si, sd, sk = downscale(c1, d1, K, 5-i)

        # plt.subplot(2, 4, i)
        # plt.imshow(si)
        # plt.title('scale level = {}'.format(i))
        # plt.subplot(2, 4, i + 4)
        # plt.imshow(sd, cmap='gray')
        # plt.title('depth(scale={})'.format(i))
    # fig.savefig('{}/downscale-a.png'.format(output_dir))

    logV('taskA is finished.....')


def method01(K, colors, depths, t1, input_dir='./data', output_dir='./output', batch_size=500, lvl=5):
    timestamp = t1
    pose_estimate, kf_estimate, distance_trans, delta_xi = [], [], [], []
    start, idx_kf = 0, 0

    for i in np.arange(start, batch_size):
        if i == 0:
            last_keyframe_pose = np.identity(4)
            c1 = np.double(imReadByGray('{}/{}'.format(input_dir, colors[i])))
            d1 = np.double(imReadByGray('{}/{}'.format(input_dir, depths[i]))) / 5000
            ckf, dkf = c1, d1

        # compute the reference frame with the keyframe
        c2 = np.double(imReadByGray('{}/{}'.format(input_dir, colors[i])))
        d2 = np.double(imReadByGray('{}/{}'.format(input_dir, depths[i]))) / 5000
        xi, errors, _ = doAlignment(ref_img=ckf, ref_depth=dkf, target_img=c2, target_depth=d2,
                                    k=K, scaled_level=lvl)
        delta_xi.append(xi)
        logD('{} | {:04d} -> xi: {}'.format('method-01', i + 1, ['%-.08f' % x for x in xi]))

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
        pose_estimate.append(result)
        logV('{} | pose({:04d} -> {:04d}) =\n\t{}'.format('method-01', i, idx_kf, result))

    if len(pose_estimate) > 0:
        np.save('{}/delta_xs-{}'.format(output_dir, lvl), delta_xi)
        np.save('{}/pose_estimate_1-{}'.format(output_dir, lvl), pose_estimate)
        np.save('{}/kf_estimate_1-{}'.format(output_dir, lvl), kf_estimate)
        np.save('{}/distance_trans-{}'.format(output_dir, lvl), distance_trans)
        saveData(pose_estimate, outdir=output_dir, fn='pose_estimate_1-{}.txt'.format(lvl))
        saveData(pose_estimate, outdir=output_dir, fn='kf_estimate_1-{}.txt'.format(lvl))
    return delta_xi, pose_estimate, distance_trans


def method02(K, colors, depths, t1, input_dir='./data', output_dir='./output',
             batch_size=500, d=0.052, a=0.012, lvl=5):
    timestamp = t1
    pose_estimate, kf_estimate, delta_xi = [], [], []
    distance_trans = []

    start, idx_kf, need_kf = 0, 0, 0
    for i in np.arange(start, batch_size):
        if i == 0:
            last_keyframe_pose = np.identity(4)
            c1 = np.double(imReadByGray('{}/{}'.format(input_dir, colors[i])))
            d1 = np.double(imReadByGray('{}/{}'.format(input_dir, depths[i]))) / 5000
            ckf, dkf = c1, d1

        # compute the reference frame with the keyframe
        c2 = np.double(imReadByGray('{}/{}'.format(input_dir, colors[i])))
        d2 = np.double(imReadByGray('{}/{}'.format(input_dir, depths[i]))) / 5000
        xi, errors, _ = doAlignment(ref_img=ckf, ref_depth=dkf, target_img=c2, target_depth=d2,
                                    k=K, scaled_level=lvl)
        delta_xi.append(xi)
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
        pose_estimate.append(result)
        logV('{} | pose({:04d} -> {:04d}) = {:.06f}\n\t{}'.format('method-02', i, idx_kf, distance, result))

    if len(pose_estimate) > 0:
        np.save('{}/delta_xs_2-{}'.format(output_dir, lvl), delta_xi)
        np.save('{}/pose_estimate_2-{}'.format(output_dir, lvl), pose_estimate)
        np.save('{}/kf_estimate_2-{}'.format(output_dir, lvl), kf_estimate)
        np.save('{}/distance_trans-{}'.format(output_dir, lvl), distance_trans)
        saveData(pose_estimate, outdir=output_dir, fn='pose_estimate_2-{}.txt'.format(lvl))
        saveData(kf_estimate, outdir=output_dir, fn='kf_estimate_2-{}.txt'.format(lvl))
    return delta_xi, pose_estimate, distance_trans


def method03(K, colors, depths, t1, input_dir='./data', output_dir='./output', batch_size=500, threshold=.9, lvl=5):
    timestamp = t1
    kf_estimate, delta_xi = [], []
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
        xi, errors, H_xi = doAlignment(ref_img=ckf, ref_depth=dkf, target_img=c2, target_depth=d2,
                                       k=K, scaled_level=lvl)
        delta_xi.append(xi)
        logD('{:04d} -> xi: {}'.format(i, ['%-.08f' % x for x in xi]))

        if i == 0:
            H_xi_t0 = H_xi

        # compute relative transform matrix
        t_inverse = inv(se3Exp(xi))  # just make sure current frame to keyframe

        if i == key_frame_index + 1:
            H_xi_t0 = H_xi

        # entropy ratio, save entropy of all images
        alpha = H_xi / H_xi_t0
        need_kf = need_kf + 1
        if alpha < threshold or need_kf >= 15:
            # here just choose the keyframe & update keyframe
            last_keyframe_pose = last_keyframe_pose @ t_inverse
            ckf, dkf = c2, d2
            key_frame_index = i
            kf_idx.append(i)
            logD('keyframe_index: {}\n\tctx: {}'.format(i, last_keyframe_pose))
            H_xi_t0 = H_xi
            alpha = H_xi / H_xi_t0
            # change the pose of last keyframe to new format, and add to list
            R = last_keyframe_pose[:3, :3]  # rotation matrix
            t = last_keyframe_pose[:3, 3]  # t
            q = Rotation.from_matrix(R).as_quat()
            tmp = np.concatenate(([timestamp[i]], t, q))
            kf = [eval('%-.08f' % x) for x in tmp]
            kf_estimate.append(kf)
            need_kf = 0
            logV('{} | ({:04d} -> {}) result: {}'.format('method-03', i, key_frame_index, kf))

        entropy_ratio.append(alpha)
        logV('{} | entropy of ({:04d} -> {:04d}) = {}'.format('method-03', i, key_frame_index, alpha))

    if len(kf_estimate) > 0:
        np.save('{}/kf_estimate_3-{}'.format(output_dir, lvl), kf_estimate)
        np.save('{}/entropy_rate-{}'.format(output_dir, lvl), entropy_ratio)
        np.save('{}/kf_idx-{}'.format(output_dir, lvl), kf_idx)
        saveData(kf_estimate, outdir=output_dir, fn='kf_estimate_3-{}.txt'.format(lvl))
    return kf_estimate, entropy_ratio, kf_idx


def taskD(K, input_dir, keyframes, kf_idx, rgbs, depths, t1):
    kfd, deltas, kf_errors = [], [], []
    # (d) optimization of keyframe pose
    for i, (kf_i, idx) in enumerate(zip(keyframes, kf_idx.astype(np.int))):
        kj, ej = [], []
        for j in [i - 1, i, i + 1]:
            if j < 0 or j > len(keyframes) - 1:
                j = (j + len(keyframes)) % len(keyframes)
            kf_j = keyframes[j]
            oidx = kf_idx.astype(np.int)[j]
            ref_i, ref_d, t_i, t_d = rgbs[idx], depths[idx], rgbs[oidx], depths[oidx]
            c1 = np.double(imReadByGray('{}/{}'.format(input_dir, ref_i)))
            d1 = np.double(imReadByGray('{}/{}'.format(input_dir, ref_d))) / 5000
            c2 = np.double(imReadByGray('{}/{}'.format(input_dir, t_i)))
            d2 = np.double(imReadByGray('{}/{}'.format(input_dir, t_d))) / 5000
            xi, errors, _ = doAlignment(ref_img=c1, ref_depth=c2, target_img=c2, target_depth=d2, k=K)
            kf_pose = np.identity(4)
            t_inverse = inv(se3Exp(xi))
            kf = kf_pose @ t_inverse
            T1, T2, rij = relativeError(kf_ref=kf_i, kf=kf_j, delta=kf)

            # Gaussian-Newton
            xi = poseGraph(c1, d1, c2, d2, K, rij)
            t_inverse = inv(se3Exp(xi))
            T1 = T1 @ t_inverse
            R, t = T1[:3, :3], T1[:3, 3]
            q = Rotation.from_matrix(R).as_quat()
            kf = np.concatenate(([kf_i[0]], t, q))
            kf = [eval('{:08f}'.format(x)) for x in kf]
            kj.append(kf)
        kj = np.asarray(kj)
        kf_i[1:] / np.mean(kj[:,1:], axis=0)
        # kfd.append(kf)
        # deltas.append(d)
        # kf_errors.append(e)

    return kfd, deltas, kf_errors


def taskE(K, input_dir, keyframes, kf_idx, rgbs, depths, t1):

    pass


def translation_distance(relative_pose):
    t = relative_pose[:3, 3]
    return np.sqrt((np.sum(t ** 2)))
