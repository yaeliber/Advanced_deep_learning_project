import os
import cv2
import random
import numpy as np
import ot as ot
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


def ArrayToKeyPoints(arr):
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


def sinkhorn_match(desc1, desc2):
    dustbin_percentage = 0.4
    len1 = len(desc1)
    len2 = len(desc2)
    cost_matrix = np.empty((len1 + 1, len2 + 1), dtype=float)
    print(cost_matrix.shape)

    # fill the cost matrix by the distance between the descriptors
    for i in range(len1):
        for j in range(len2):
            cost_matrix[i][j] = np.linalg.norm(desc1[i] - desc2[j])  # L2

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

    # print('a len: ', len(a))
    # print('a sum: ', np.sum(a))  # should be 1
    # print('\nb len: ', len(b))
    # print('b sum: ', np.sum(b))  # should be 1

    res = ot.sinkhorn(a, b, cost_matrix, 10, method='sinkhorn_stabilized')
    # print(res)
    print('res shape', res.shape)

    max_index_arr = np.argmax(res, axis=1)
    # print('max_index_arr: ', max_index_arr)

    # temp = np.take_along_axis(res, np.expand_dims(max_index_arr, axis=-1), axis=-1).squeeze(axis=-1)
    # print('temp: ', temp)

    match = []
    for i in range(len1):
        if max_index_arr[i] == len2:
            continue
        dist = np.linalg.norm(desc1[i] - desc2[max_index_arr[i]])
        match.append(cv2.DMatch(i, max_index_arr[i], dist))

    print('best_matches_sinkhorn: ', len(match))
    return match


def find_homography(img1, img2, kp1, kp2, best_matches, algorithm=''):
    print('--------- In find_homography ---------')
    src_pts = np.float32([kp1[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
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
    print('--------- In print_wraped_images ---------')
    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(1, 3, 1)
    plt.title('img1')
    plt.axis('off')
    plt.imshow(img1)

    fig.add_subplot(1, 3, 2)
    plt.title('img2_warped')
    plt.axis('off')
    plt.imshow(img2_warped)

    fig.add_subplot(1, 3, 3)
    plt.title('img2')
    plt.axis('off')
    plt.imshow(img2)
    plt.show()
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
    kp1 = ArrayToKeyPoints(data['kp1'])
    kp2 = ArrayToKeyPoints(data['kp2'])

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
        best_matches = sinkhorn_match(desc1, desc2)

    if len(best_matches) < 4:
        return None, 0, 50, 50, 10

    H, mask, img2_warped = find_homography(img1, img2, kp1, kp2, best_matches, algorithm)

    if H is None:
        return None, 0, 50, 50, 10

    match_score = get_match_score(kp1, kp2, best_matches, data['M'], data['I'], data['J'])

    error_H, H_mean, H_std = H_error(H, path3)

    # print_wraped_images(img1, img2, img2_warped)

    return H, match_score, error_H, H_mean, H_std


def get_match_score(kp1, kp2, best_matches, M, I, J):
    print('--------- In get_match_score ---------')
    # extract keyPoints from params we made on dataSetCreate
    m_source = ArrayToKeyPoints(M[0])
    m_dest = ArrayToKeyPoints(M[1])

    M = [m_source, m_dest]

    src_pts = np.float32([kp1[m.queryIdx].pt for m in best_matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in best_matches])

    M_ = [src_pts, dst_pts]
    I_ = [item for item in kp1 if item.pt not in src_pts]
    J_ = [item for item in kp2 if item.pt not in dst_pts]

    M_counter = 0
    print('len M  ', len(M[0]))
    print('len M* ', len(M_[0]))
    for j in range(len(M_[0])):
        for i in range(len(M[0])):
            if M[0][i].pt[0] == M_[0][j][0] and M[0][i].pt[1] == M_[0][j][1] \
                    and M[1][i].pt[0] == M_[1][j][0] and M[1][i].pt[1] == M_[1][j][1]:
                M_counter += 1
                break
    I = ArrayToKeyPoints(I)
    print('len I  ', len(I))
    print('len I* ', len(I_))
    I_counter = 0
    for kp_1 in I_:
        for kp_2 in I:
            if kp_1.pt[0] == kp_2.pt[0] and kp_1.pt[1] == kp_2.pt[1]:
                I_counter += 1
                break

    J = ArrayToKeyPoints(J)
    print('len J  ', len(J))
    print('len J* ', len(J_))
    J_counter = 0
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

    H_mean, H_std = getDifficultLevel(H_src_to_dest)

    return error, H_mean, H_std


def getDifficultLevel(H):
    I = np.eye(3)
    dif = np.abs(H - I)
    H_mean = np.mean(dif)
    H_std = np.std(dif)
    return H_mean, H_std


def main(folder_path, folder_number):
    error_H_sinkhorn = []
    error_H_knn = []
    error_H_knn_v2 = []
    mean_H = []
    match_score_sinkhorn = []
    match_score_knn = []
    match_score_knn_v2 = []
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

        H2_dest_to_src, match_score2, error_H2, H_mean, H_std = make_match(path1, path2, path3, 'knn_match')
        error_H_knn.append(error_H2)
        match_score_knn.append(match_score2)

        H2_dest_to_src, match_score3, error_H3, H_mean, H_std = make_match(path1, path2, path3, 'knn_match_v2')
        error_H_knn_v2.append(error_H3)
        match_score_knn_v2.append(match_score3)
        print()

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.title('error_H')
    plt.plot(error_H_sinkhorn, 'or', label='sinkhorn')
    plt.plot(error_H_knn, 'ob', label='knn')
    plt.plot(error_H_knn_v2, 'og', label='knn_v2')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title('H mean difficult')
    plt.plot(mean_H, 'ob')

    plt.figure(figsize=(10, 10))
    plt.title('match_score')
    ax = plt.gca()
    ax.set_ylim([0, 1])
    plt.plot(match_score_sinkhorn, 'or', label='sinkhorn')
    plt.plot(match_score_knn, 'ob', label='knn')
    plt.plot(match_score_knn_v2, 'og', label='knn_v2')
    plt.legend()

    mean_MIJ_score = []
    mean_MIJ_score.append(np.sum(match_score_sinkhorn) / len(match_score_sinkhorn))
    mean_MIJ_score.append(np.sum(match_score_knn) / len(match_score_knn))
    mean_MIJ_score.append(np.sum(match_score_knn_v2) / len(match_score_knn_v2))
    plt.figure(figsize=(5, 5))
    labels = ['sinkhorn', 'knn', 'knn_v2']
    ax = plt.gca()
    ax.set_ylim([0, 1])
    plt.bar(labels, mean_MIJ_score, width=0.4)
    plt.show()


if __name__ == '__main__':
    # folder_path = './data/resize_photos/'
    folder_path = '../../data/resize_photos/'
    folder_number = 1
    main(folder_path, folder_number)

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
