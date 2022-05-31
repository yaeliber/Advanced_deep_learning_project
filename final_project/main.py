import os
import cv2
import numpy as np
import ot as ot
import torch
import tensorflow as tf
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F

from imp_sinkhorn import *
from superGlueSinkhorn import *


# return: list of keypoints objects
def array_to_key_points(arr):
    kp = []
    for k in arr:
        kp.append(cv2.KeyPoint(k[0][0], k[0][1], k[1], k[2], k[3], k[4], k[5]))
    return kp


def knn_match(desc1, desc2, flag=True):
    if flag:
        print('--------- In knn_match_v2 ---------')
    else:
        print('--------- In knn_match ---------')
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)

    # store all the good matches as per Lowe's ratio test.
    # distance L2
    best_matches = []
    for m, n in matches:  # for every descriptor, take closest two matches
        if flag:
            if m.distance < 0.8 * n.distance:  # best match has to be this much closer than second best
                best_matches.append(m)
        else:
            best_matches.append(m)

    print('best_matches_knn: ', len(best_matches))
    print('\n\n')

    return best_matches


def linear_assignment_match(desc1, desc2):
    len1 = len(desc1)
    len2 = len(desc2)
    cost_matrix = np.empty((len1, len2), dtype=float)
    print(cost_matrix.shape)
    # fill the cost matrix by the distance between the descriptors
    for i in range(len1):
        for j in range(len2):
            cost_matrix[i][j] = np.linalg.norm(desc1[i] - desc2[j])  # L2

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    match = []
    for i in range(len(row_ind)):
        dist = np.linalg.norm(desc1[row_ind[i]] - desc2[col_ind[i]])
        if dist < 200:
            match.append(cv2.DMatch(row_ind[i], col_ind[i], dist))
    print('best_matches_linear_assignment: ', len(match))
    return match


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


def sinkhorn_match2(desc1, desc2, dp_percentage):
    print("on sinkhorn_match2")
    len1 = len(desc1)
    len2 = len(desc2)
    # normalize the descriptors
    # norms1 = torch.linalg.norm(desc1, dim=1, ord=2)+0.01
    # norms2 = torch.linalg.norm(desc2, dim=1, ord=2)+0.01
    # d1 = torch.div(desc1.T, norms1).T
    # d2 = torch.div(desc2.T, norms2).T
    d1 = torch.reshape(desc1, (1, desc1.shape[0], 128))
    d2 = torch.reshape(desc2, (1, desc2.shape[0], 128))
    # fill the cost matrix by the inner dot between the descriptors
    cost_matrix = torch.einsum('bnd,bmd->bnm', d1, d2)
    # cost_matrix = cost_matrix / (128 ** 0.5)

    # print('cost_matrix', cost_matrix)
    res = log_optimal_transport(cost_matrix, dp_percentage, iters=700)
    # print("line 96 res", res)
    # max_index_arr = torch.argmax(res[0], axis=1)

    # Get the matches with score above "match_threshold".
    max0, max1 = res[:, :-1, :-1].max(2), res[:, :-1, :-1].max(1)
    indices0, indices1 = max0.indices, max1.indices
    mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
    mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
    zero = res.new_tensor(0)
    mscores0 = torch.where(mutual0, max0.values.exp(), zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
    valid0 = mutual0 & (mscores0 > 0.2)
    valid1 = mutual1 & valid0.gather(1, indices1)
    indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
    indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

    match = []
    for i in range(len1):
        # print("indices0 ", indices0[0])
        # print("indices0[i] ", indices0[0][i])
        if indices0[0][i] == -1:  # if matched to dustbin
            continue
        dist = torch.floor(torch.linalg.norm(desc1[i] - desc2[indices0[0][i]]))
        if dist != dist:  # dist is nan
            dist = torch.zeros(1)
        match.append(cv2.DMatch(i, indices0[0][i].item(), int(dist.item())))

    # match = []
    # for i in range(len1):
    #     if max_index_arr[i] == len2:  # if matched to dustbin
    #         continue
    #     dist = torch.floor(torch.linalg.norm(desc1[i] - desc2[max_index_arr[i]]))
    #     if dist != dist:  # dist is nan
    #         dist = torch.zeros(1)
    #     match.append(cv2.DMatch(i, max_index_arr[i].item(), int(dist.item())))

    return res[0], match


def sinkhorn_match(desc1, desc2, dp_percentage=0.4):
    dustbin_percentage = dp_percentage
    len1 = len(desc1)
    len2 = len(desc2)
    cost_matrix = torch.empty((len1 + 1, len2 + 1), dtype=float)
    # print(cost_matrix.shape)

    # fill the cost matrix by the distance between the descriptors
    for i in range(len1):
        for j in range(len2):
            cost_matrix[i][j] = torch.linalg.norm(desc1[i] - desc2[j])  # L2
            # cost_matrix[i][j] = torch.dot(desc1[i],desc2[j]).item()/(128**0.5)

    # fill the dustbin rows and cols to 0
    for i in range(len1 + 1):
        cost_matrix[i][len2] = 0
    for j in range(len2 + 1):
        cost_matrix[len1][j] = 0

    # unify distribution beside dustbin cell
    a = [(1 - dustbin_percentage) / len1] * (len1 + 1)
    a[len1] = dustbin_percentage

    # unify distribution beside dustbin cell
    b = [(1 - dustbin_percentage) / len2] * (len2 + 1)
    b[len2] = dustbin_percentage

    a = torch.Tensor(a)
    b = torch.Tensor(b)
    res = ot.sinkhorn(a, b, cost_matrix, 10, method='sinkhorn_stabilized')
    # print("line 96 res", res)
    max_index_arr = torch.argmax(res, axis=1)

    match = []
    for i in range(len1):
        if max_index_arr[i] == len2:  # if matched to dustbin
            continue
        dist = torch.floor(torch.linalg.norm(desc1[i] - desc2[max_index_arr[i]]))
        match.append(cv2.DMatch(i, max_index_arr[i].item(), int(dist.item())))

    return res, match


def find_homography(img1, img2, kp1, kp2, best_matches, algorithm=''):
    print('--------- In find_homography ---------')
    src_pts = np.float32([kp1[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)  # None to min squres
    print('H: ', H)
    if H is None:
        return None, None, None
    matchesMask = mask.ravel().tolist()
    draw_params = dict(singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, best_matches, None, **draw_params)
    # fig = plt.figure(figsize=(10, 10))
    # fig.suptitle(algorithm)
    # plt.title('keypoints matches')
    # plt.axis('off')
    # plt.imshow(img3)
    # plt.show()

    img2_warped = cv2.warpPerspective(img2, H, (img2.shape[1], img2.shape[0]))
    print('\n\n')
    return H, mask, img2_warped


def print_wraped_images(img1, img2, img2_warped):
    # print('--------- In print_wraped_images ---------')
    # fig = plt.figure(figsize=(10, 10))
    # fig.add_subplot(1, 3, 1)
    # plt.title('img1')
    # plt.axis('off')
    # plt.imshow(img1)
    #
    # fig.add_subplot(1, 3, 2)
    # plt.title('img2_warped')
    # plt.axis('off')
    # plt.imshow(img2_warped)
    #
    # fig.add_subplot(1, 3, 3)
    # plt.title('img2')
    # plt.axis('off')
    # plt.imshow(img2)
    # plt.show()
    print('\n\n')


def make_match(path1, path2, path3, algorithm):
    img1 = cv2.cvtColor(cv2.imread(path1), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(path2), cv2.COLOR_BGR2RGB)
    data = np.load(path3, allow_pickle=True)

    # fig = plt.figure(figsize=(10, 10))
    # fig.add_subplot(1, 2, 1)
    # plt.axis('off')
    # plt.imshow(img1)
    #
    # fig.add_subplot(1, 2, 2)
    # plt.axis('off')
    # plt.imshow(img2)
    # plt.show()

    # extract keyPoints from params we made on dataSetCreate
    kp1 = array_to_key_points(data['kp1'])
    kp2 = array_to_key_points(data['kp2'])

    print('kp1: ', len(kp1))
    print('kp2: ', len(kp2))

    desc1, desc2 = data['desc1'], data['desc2']

    if algorithm == 'knn_match':  # all knn matches without
        best_matches = knn_match(desc1, desc2, False)
    if algorithm == 'knn_match_v2':
        best_matches = knn_match(desc1, desc2, True)
    if algorithm == 'linear_assignment_match':
        best_matches = linear_assignment_match(desc1, desc2)
    if algorithm == 'sinkhorn_match':
        __, best_matches = sinkhorn_match(torch.as_tensor(desc1), torch.as_tensor(desc2), 0.4)
    if algorithm == 'sinkhorn_match2':
        __, best_matches = sinkhorn_match2(torch.as_tensor(desc1), torch.as_tensor(desc2), torch.ones(1) * 0.4)

    if len(best_matches) < 4:
        return None, 0, 50, 50, 10

    H, mask, img2_warped = find_homography(img1, img2, kp1, kp2, best_matches, algorithm)

    if H is None:
        return None, 0, 50, 50, 10

    match_score = get_match_score(kp1, kp2, best_matches, data['M'], data['I'], data['J'])

    error_H, H_mean, H_std = H_error(H, path3)
    if algorithm == 'sinkhorn_match':
        print_wraped_images(img1, img2, img2_warped)

    return H, match_score, error_H, H_mean, H_std


def get_match_score(kp1, kp2, best_matches, M, I, J):
    print('--------- In get_match_score ---------')
    # extract keyPoints from params we made on dataSetCreate
    m_source = array_to_key_points(M[0])
    m_dest = array_to_key_points(M[1])

    M = [m_source, m_dest]

    src_pts = np.float32([kp1[m.queryIdx].pt for m in best_matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in best_matches])

    M_ = [src_pts, dst_pts]
    I_ = [item for item in kp1 if item.pt not in src_pts]
    J_ = [item for item in kp2 if item.pt not in dst_pts]

    print('len M  ', len(M[0]))
    print('len M* ', len(M_[0]))
    M_counter = 0
    # M_counter is counting intersection between M and M_ (M_ = M*)
    for j in range(len(M_[0])):
        for i in range(len(M[0])):
            if M[0][i].pt[0] == M_[0][j][0] and M[0][i].pt[1] == M_[0][j][1] \
                    and M[1][i].pt[0] == M_[1][j][0] and M[1][i].pt[1] == M_[1][j][1]:
                M_counter += 1
                break
    I = array_to_key_points(I)
    print('len I  ', len(I))
    print('len I* ', len(I_))
    I_counter = 0
    # I_counter is counting intersection between I and I_ (I_ = I*)
    for kp_1 in I_:
        for kp_2 in I:
            if kp_1.pt[0] == kp_2.pt[0] and kp_1.pt[1] == kp_2.pt[1]:
                I_counter += 1
                break

    J = array_to_key_points(J)
    print('len J  ', len(J))
    print('len J* ', len(J_))
    J_counter = 0
    # J_counter is counting intersection between J and J_ (J_ = J*)
    for kp_1 in J_:
        for kp_2 in J:
            if kp_1.pt[0] == kp_2.pt[0] and kp_1.pt[1] == kp_2.pt[1]:
                J_counter += 1
                break

    print('-----------------')
    print('M_counter: ', M_counter)
    print('I_counter: ', I_counter)
    print('J_counter: ', J_counter)
    score = (M_counter + I_counter + J_counter) / (len(M[0]) + len(I) + len(J))
    print('match score: ', score)

    return score


def H_error(H_dest_to_src, path):
    # the func return the distance of H.dot(H*) from I
    data = np.load(path, allow_pickle=True)
    H_src_to_dest = data['H']  # Homograpy matrix from src to dest
    error = H_src_to_dest.dot(H_dest_to_src) - np.eye(3)
    error = np.sum(np.abs(error))

    H_mean, H_std = get_difficult_level(H_src_to_dest)

    return error, H_mean, H_std


def get_difficult_level(H):
    I = np.eye(3)
    dif = np.abs(H - I)
    H_mean = np.mean(dif)
    H_std = np.std(dif)
    return H_mean, H_std


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------         2 Level Matching         ------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def intersection_match(kp1, kp2, best_matches1, best_matches2):
    intersection_best_match = []

    src_pts1 = [kp1[m.queryIdx].pt for m in best_matches1]
    dst_pts1 = [kp2[m.trainIdx].pt for m in best_matches1]

    src_pts2 = [kp1[m.queryIdx].pt for m in best_matches2]
    dst_pts2 = [kp2[m.trainIdx].pt for m in best_matches2]

    for index1, src in enumerate(src_pts1):
        list_index2 = [i for i, x in enumerate(src_pts2) if x == src]
        for i in list_index2:
            if dst_pts1[index1] == dst_pts2[i]:
                intersection_best_match.append(best_matches1[index1])

    return intersection_best_match


def multy_level_match(kp1, kp2, desc1, desc2, algorithm1, algorithm2):
    kp11 = []
    kp22 = []
    desc11 = []
    desc22 = []
    best_matches = []
    best_matches2 = []

    # first algorithm
    if algorithm1 == 'knn_match':  # all knn matches without
        best_matches = knn_match(desc1, desc2, False)
    if algorithm1 == 'knn_match_v2':
        best_matches = knn_match(desc1, desc2, True)
    if algorithm1 == 'linear_assignment_match':
        best_matches = linear_assignment_match(desc1, desc2)
    if algorithm1 == 'sinkhorn_match':
        __, best_matches = sinkhorn_match(torch.as_tensor(desc1), torch.as_tensor(desc2), 0.4)
    if algorithm1 == 'sinkhorn_match2':
        __, best_matches = sinkhorn_match2(torch.as_tensor(desc1), torch.as_tensor(desc2), torch.ones(1) * 0.4)

    # build new keypoints and descriptors thar are not matched
    src_pts = np.float32([kp1[m.queryIdx].pt for m in best_matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in best_matches])

    for index, item in enumerate(kp1):
        if item.pt not in src_pts:
            kp11.append(item)
            desc11.append(desc1[index])

    for index, item in enumerate(kp2):
        if item.pt not in dst_pts:
            kp22.append(item)
            desc22.append(desc2[index])

    # second algorithm
    if algorithm2 == 'knn_match':  # all knn matches without
        best_matches2 = knn_match(desc11, desc22, False)
    if algorithm2 == 'knn_match_v2':
        best_matches2 = knn_match(desc11, desc22, True)
    if algorithm2 == 'linear_assignment_match':
        best_matches2 = linear_assignment_match(desc11, desc22)
    if algorithm2 == 'sinkhorn_match':
        __, best_matches2 = sinkhorn_match(torch.as_tensor(desc11), torch.as_tensor(desc22), 0.4)
    if algorithm2 == 'sinkhorn_match2':
        __, best_matches2 = sinkhorn_match2(torch.as_tensor(desc11), torch.as_tensor(desc22), torch.ones(1) * 0.4)

    len_best_matches1 = len(best_matches1)
    # extend the best matches of the two algorithms
    for match in  best_matches2:
        match.queryIdx += len_best_matches1
        match.trainIdx += len_best_matches1

    return kp1.extend(kp11), kp2.extend(kp22), best_matches.extend(best_matches2)


# ----------------------------------------------------------------------------------------------------------------------
def main(folder_path, folder_number):
    error_H_sinkhorn = []
    error_H_sinkhorn2 = []
    error_H_knn = []
    error_H_knn_v2 = []
    error_H_linear_assignment = []

    mean_H = []
    match_score_sinkhorn = []
    match_score_sinkhorn2 = []
    match_score_knn = []
    match_score_knn_v2 = []
    match_score_linear_assignment = []
    assert (os.path.exists(folder_path))
    for file in os.scandir(folder_path):
        file_name = file.name
        print('\n================================ ', file_name, ' ================================')
        path1 = '../../data/resize_photos/' + file_name
        path2 = '../../data/homography_photos/' + str(folder_number) + '/' + file_name
        path3 = '../../data/params/' + str(folder_number) + '/' + file_name + '.npz'
        H1_dest_to_src, match_score1, error_H1, H_mean, H_std = make_match(path1, path2, path3, 'sinkhorn_match')
        error_H_sinkhorn.append(error_H1)
        match_score_sinkhorn.append(match_score1)
        mean_H.append(H_mean)

        H1_dest_to_src, match_score1, error_H1, H_mean, H_std = make_match(path1, path2, path3, 'sinkhorn_match2')
        error_H_sinkhorn2.append(error_H1)
        match_score_sinkhorn2.append(match_score1)

        H2_dest_to_src, match_score2, error_H2, H_mean, H_std = make_match(path1, path2, path3, 'knn_match')
        error_H_knn.append(error_H2)
        match_score_knn.append(match_score2)

        H2_dest_to_src, match_score3, error_H3, H_mean, H_std = make_match(path1, path2, path3, 'knn_match_v2')
        error_H_knn_v2.append(error_H3)
        match_score_knn_v2.append(match_score3)

        H1_dest_to_src, match_score1, error_H1, H_mean, H_std = make_match(path1, path2, path3, 'linear_assignment_match')
        error_H_linear_assignment.append(error_H1)
        match_score_linear_assignment.append(match_score1)
        print()

    fig = plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.title('error_H')
    plt.plot(error_H_sinkhorn, 'or', label='sinkhorn')
    plt.plot(error_H_sinkhorn2, 'oc', label='sinkhorn2')
    plt.plot(error_H_knn, 'ob', label='knn')
    plt.plot(error_H_knn_v2, 'og', label='knn_v2')
    plt.plot(error_H_linear_assignment, 'ok', label='linear_assignment_match')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title('H mean difficult')
    plt.plot(mean_H, 'ob')
    fig.savefig('../../data/graphs/errorH.png')

    fig = plt.figure(figsize=(10, 10))
    plt.title('match_score')
    ax = plt.gca()
    ax.set_ylim([0, 1])
    plt.plot(match_score_sinkhorn, 'or', label='sinkhorn')
    plt.plot(match_score_sinkhorn2, 'oc', label='sinkhorn2')
    plt.plot(match_score_knn, 'ob', label='knn')
    plt.plot(match_score_knn_v2, 'og', label='knn_v2')
    plt.plot(match_score_linear_assignment, 'ok', label='linear_assignment_match')
    plt.legend()
    fig.savefig('../../data/graphs/MIJscore.png')

    # A graph that shows the MIJ_score average according to each algorithm
    mean_MIJ_score = []
    mean_MIJ_score.append(np.sum(match_score_sinkhorn) / len(match_score_sinkhorn))
    mean_MIJ_score.append(np.sum(match_score_sinkhorn2) / len(match_score_sinkhorn2))
    mean_MIJ_score.append(np.sum(match_score_knn) / len(match_score_knn))
    mean_MIJ_score.append(np.sum(match_score_knn_v2) / len(match_score_knn_v2))
    mean_MIJ_score.append(np.sum(match_score_linear_assignment) / len(match_score_linear_assignment))
    fig = plt.figure(figsize=(5, 5))
    plt.title('mean_match_score')
    labels = ['sinkhorn', 'sinkhorn2', 'knn', 'knn_v2', 'linear_assignment']
    ax = plt.gca()
    ax.set_ylim([0, 1])
    plt.bar(labels, mean_MIJ_score, width=0.4)
    fig.savefig('../../data/graphs/meanMatchScore.png')

    # A graph that shows the H_error average according to each algorithm
    mean_H_error = []
    mean_H_error.append(np.sum(error_H_sinkhorn) / len(error_H_sinkhorn))
    mean_H_error.append(np.sum(error_H_sinkhorn2) / len(error_H_sinkhorn2))
    mean_H_error.append(np.sum(error_H_knn) / len(error_H_knn))
    mean_H_error.append(np.sum(error_H_knn_v2) / len(error_H_knn_v2))
    mean_H_error.append(np.sum(error_H_linear_assignment) / len(error_H_linear_assignment))
    fig = plt.figure(figsize=(5, 5))
    plt.title('mean_H_error')
    labels = ['sinkhorn', 'sinkhorn2', 'knn', 'knn_v2', 'linear_assignment']
    plt.bar(labels, mean_H_error, width=0.4)
    fig.savefig('../../data/graphs/meanHScore.png')

    plt.show()


if __name__ == '__main__':
    folder_path = '../../data/resize_photos/'
    # folder_path = '../../data/test/'
    folder_number = 1
    main(folder_path, folder_number)
    # kp1 = [{"pt": (1, 7)}, {"pt": (2, 3)}, {"pt": (5, 5)}, {"pt": (9, 0)}, {"pt": (1, 1)}]
    # kp2 = [{"pt": (5, 4)}, {"pt": (2, 4)}, {"pt": (6, 7)}, {"pt": (8, 8)}, {"pt": (9, 5)}]
    # best_matches1 = [{"queryIdx": 0, "trainIdx": 2}, {"queryIdx": 1, "trainIdx": 1}, {"queryIdx": 1, "trainIdx": 3}]
    # best_matches2 = [{"queryIdx": 4, "trainIdx": 2}, {"queryIdx": 1, "trainIdx": 1}, {"queryIdx": 1, "trainIdx": 3}]
    # print(intersection_match(kp1, kp2, best_matches1, best_matches2))
    # =================================================================================================================

    # file_name = 'paris.jpg'
    # path1 = './data/resize_photos/' + file_name
    # path2 = './data/homography_photos/2/' + file_name
    # path3 = './data/params/2/' + file_name + '.npz'
    # H1_dest_to_src, match_score1, error_H1, H_mean, H_std = make_match(path1, path2, path3, 'sinkhorn_match')
    # H2_dest_to_src, match_score2, error_H2, H_mean, H_std = make_match(path1, path2, path3, 'knn_match')

    # =================================================================================================================

    # path2 = './data/homography_photos/1/' + file_name
    # path3 = './data/params/1/' + file_name + '.npz'
    # H_dest_to_src = make_match(path1, path2, path3)
    # error = H_error(H_dest_to_src, path3)
    # print('error: ', error)
