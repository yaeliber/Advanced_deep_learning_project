import os
import cv2
import random
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


def ArrayToKeyPoints(arr):
    kp = []
    for k in arr:
        kp.append(cv2.KeyPoint(k[0][0], k[0][1], k[1], k[2], k[3], k[4], k[5]))
    return kp


def get_best_matches(desc1, desc2):
    print("--------- In get_best_matches ---------")
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # print("index_params:", index_params)
    search_params = dict(checks=50)
    # print("search_params:", search_params)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # print(flann)
    matches = flann.knnMatch(desc1, desc2, k=2)
    # print("len of matches:", len(matches))
    # print("matches:", matches)

    # store all the good matches as per Lowe's ratio test.
    # distance L2
    best_matches = []
    for m, n in matches:  # for every descriptor, take closest two matches
        if m.distance < 0.7 * n.distance:  # best match has to be this much closer than second best
            best_matches.append(m)

    print(len(best_matches))
    print("\n\n")

    return best_matches


def find_homography(img1, img2, kp1, kp2, best_matches):
    print("--------- In find_homography ---------")
    src_pts = np.float32([kp1[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    matchesMask = mask.ravel().tolist()
    draw_params = dict(singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, best_matches, None, **draw_params)
    fig = plt.figure(figsize=(10, 10))
    plt.title("keypoints matches")
    plt.axis("off")
    plt.imshow(img3)
    plt.show()

    img2_warped = cv2.warpPerspective(img2, H, (img2.shape[1], img2.shape[0]))
    print("\n\n")
    return H, mask, img2_warped


def print_wraped_images(img1, img2, img2_warped):
    print("--------- In print_wraped_images ---------")
    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(1, 3, 1)
    plt.title("img1")
    plt.axis("off")
    plt.imshow(img1)

    fig.add_subplot(1, 3, 2)
    plt.title("img2_warped")
    plt.axis("off")
    plt.imshow(img2_warped)

    fig.add_subplot(1, 3, 3)
    plt.title("img2")
    plt.axis("off")
    plt.imshow(img2)
    plt.show()
    print("\n\n")


def make_match(path1, path2, path3):
    img1 = cv2.cvtColor(cv2.imread(path1), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(path2), cv2.COLOR_BGR2RGB)
    data = np.load(path3, allow_pickle=True)

    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(1, 2, 1)
    plt.axis("off")
    plt.imshow(img1)

    fig.add_subplot(1, 2, 2)
    plt.axis("off")
    plt.imshow(img2)
    plt.show()

    # extract keyPoints from params we made on dataSetCreate
    kp1 = ArrayToKeyPoints(data['kp1'])
    kp2 = ArrayToKeyPoints(data['kp2'])

    desc1, desc2 = data['desc1'], data['desc2']

    best_matches = get_best_matches(desc1, desc2)

    H, mask, img2_warped = find_homography(img1, img2, kp1, kp2, best_matches)

    match_score = get_match_score(kp1, kp2, best_matches, data['M'], data['I'], data['J'])

    print_wraped_images(img1, img2, img2_warped)

    return H


def get_match_score(kp1, kp2, best_matches, M, I, J):
    print("--------- In get_match_score ---------")
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
    print("len M  ", len(M[0]))
    print("len M* ", len(M_[0]))
    for j in range(len(M_[0])):
        for i in range(len(M[0])):
            if M[0][i].pt[0] == M_[0][j][0] and M[0][i].pt[1] == M_[0][j][1] \
                    and M[1][i].pt[0] == M_[1][j][0] and M[1][i].pt[1] == M_[1][j][1]:
                M_counter += 1
                break
    I = ArrayToKeyPoints(I)
    print("len I  ", len(I))
    print("len I* ", len(I_))
    I_counter = 0
    for kp_1 in I_:
        for kp_2 in I:
            if kp_1.pt[0] == kp_2.pt[0] and kp_1.pt[1] == kp_2.pt[1]:
                I_counter += 1
                break

    J = ArrayToKeyPoints(J)
    print("len J  ", len(J))
    print("len J* ", len(J_))
    J_counter = 0
    for kp_1 in J_:
        for kp_2 in J:
            if kp_1.pt[0] == kp_2.pt[0] and kp_1.pt[1] == kp_2.pt[1]:
                J_counter += 1
                break

    print("-----------------")
    print("M_counter: ", M_counter)
    print("I_counter: ", I_counter)
    print("J_counter: ", J_counter)
    score = (M_counter + I_counter + J_counter) / (len(M[0]) + len(I) + len(J))
    print("match score: ", score)

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


if __name__ == '__main__':
    file_name = "Eiffel.jpg"
    path1 = "./data/resize_photos/" + file_name
    path2 = "./data/homography_photos/2/" + file_name
    path3 = "./data/params/2/" + file_name + ".npz"
    H_dest_to_src = make_match(path1, path2, path3)
    error, H_mean, H_std = H_error(H_dest_to_src, path3)
    print("error: ", error)
    print("H_mean: ", H_mean)
    print("H_std: ", H_std)

    # path2 = "./data/homography_photos/2/" + file_name
    # path3 = "./data/params/2/" + file_name + ".npz"
    # H_dest_to_src = make_match(path1, path2, path3)
    # error = H_error(H_dest_to_src, path3)
    # print("error: ", error)
