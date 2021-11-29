import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np


def get_keypoints_and_descriptors(img1, img2):
    print("--------- In get_keypoints_and_descriptors ---------")
    # use orb if sift is not installed
    # feature_extractor = cv2.xfeatures2d.SIFT_create()
    feature_extractor = cv2.SIFT_create()

    # find the keypoints and descriptors with chosen feature_extractor
    kp1, desc1 = feature_extractor.detectAndCompute(img1, None)
    kp2, desc2 = feature_extractor.detectAndCompute(img2, None)
    print("img1", img1.shape)
    print("kp1", len(kp1))
    print("kp2", len(kp2))

    keyOriginal = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    keyRotated = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(1, 2, 1)
    plt.title("keyOriginalPoints")
    plt.axis("off")
    plt.imshow(keyOriginal)

    fig.add_subplot(1, 2, 2)
    plt.title("keyRotatedPoints")
    plt.axis("off")
    plt.imshow(keyRotated)
    plt.show()
    print("\n\n")

    return kp1, desc1, kp2, desc2


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
    for m, n in matches:
        if m.distance < 0.3 * n.distance:
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


def eiffle():
    path = "./temp_photos/Eiffel.jpg"

    img1 = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    gray_l = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(1, 2, 1)
    plt.axis("off")
    plt.title("original")
    plt.imshow(img1)

    fig.add_subplot(1, 2, 2)
    plt.axis("off")
    plt.title("gray")
    plt.imshow(gray_l, cmap="gray")
    plt.show()

    # rotating
    h, w = img1.shape[:2]  # 2, בלי עומק
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, 20, 1)
    img2 = cv2.warpAffine(img1, M, (w, h))

    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(1, 2, 1)
    plt.title("original")
    plt.axis("off")
    plt.imshow(img1)

    fig.add_subplot(1, 2, 2)
    plt.title("rotated")
    plt.axis("off")
    plt.imshow(img2)
    plt.show()

    kp1, desc1, kp2, desc2 = get_keypoints_and_descriptors(img1, img2)

    best_matches = get_best_matches(desc1, desc2)

    H, mask, img2_warped = find_homography(img1, img2, kp1, kp2, best_matches)

    print_wraped_images(img1, img2, img2_warped)


def room():
    path1 = "./temp_photos/room1.jpeg"
    path2 = "./temp_photos/room2.jpeg"

    room1 = cv2.cvtColor(cv2.imread(path1), cv2.COLOR_BGR2RGB)
    room2 = cv2.cvtColor(cv2.imread(path2), cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(1, 2, 1)
    plt.axis("off")
    plt.imshow(room1)

    fig.add_subplot(1, 2, 2)
    plt.axis("off")
    plt.imshow(room2)
    plt.show()

    kp1, desc1, kp2, desc2 = get_keypoints_and_descriptors(room1, room2)

    best_matches = get_best_matches(desc1, desc2)

    H, mask, img2_warped = find_homography(room1, room2, kp1, kp2, best_matches)

    print_wraped_images(room1, room2, img2_warped)


if __name__ == '__main__':
    room()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
