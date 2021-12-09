import os
import cv2
import random
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


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

    kp1 = []
    kp2 = []
    for k in data['kp1']:
        kp1.append(cv2.KeyPoint(k[0][0], k[0][1], k[1], k[2], k[3], k[4], k[5]))
    for k in data['kp2']:
        kp2.append(cv2.KeyPoint(k[0][0], k[0][1], k[1], k[2], k[3], k[4], k[5]))

    desc1, desc2 = data['desc1'], data['desc2']

    best_matches = get_best_matches(desc1, desc2)

    H, mask, img2_warped = find_homography(img1, img2, kp1, kp2, best_matches)

    print_wraped_images(img1, img2, img2_warped)


if __name__ == '__main__':
    file_name = "room2.jpeg"
    path1 = "./photos/" + file_name
    path2 = "./homography_photos/" + file_name
    path3 = "./params/" + file_name + ".npz"
    make_match(path1, path2, path3)
