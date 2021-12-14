import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt


def get_keypoints_and_descriptors(img1, img2):
    print("--------- In get_keypoints_and_descriptors ---------")
    # use orb if sift is not installed
    # feature_extractor = cv2.xfeatures2d.SIFT_create()
    feature_extractor = cv2.SIFT_create()

    # find the keypoints and descriptors with chosen feature_extractor
    kp1, desc1 = feature_extractor.detectAndCompute(img1, None)
    kp2, desc2 = feature_extractor.detectAndCompute(img2, None)
    # print("====================================================================")
    # for k in kp1:
    #     print("x = ", k.pt[0], "     y = ", k.pt[1])
    # print("====================================================================")
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


def ImagePreProcessing(path, resize_Path, homography_path, params_path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (320, 240))

    rho = 32
    patch_size = 128
    top_point = (32, 32)
    left_point = (patch_size + 32, 32)
    bottom_point = (patch_size + 32, patch_size + 32)
    right_point = (32, patch_size + 32)
    test_image = img.copy()
    four_points = [top_point, left_point, bottom_point, right_point]

    perturbed_four_points = []
    for point in four_points:
        perturbed_four_points.append(
            (point[0] + random.randint(-rho, rho), point[1] + random.randint(-rho, rho)))

    H = cv2.getPerspectiveTransform(np.float32(perturbed_four_points), np.float32(
        four_points))

    warped_image = cv2.warpPerspective(img, H, (320, 240))

    cv2.imwrite(resize_Path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(homography_path, cv2.cvtColor(
        warped_image, cv2.COLOR_RGB2BGR))

    kp1, desc1, kp2, desc2 = get_keypoints_and_descriptors(img, warped_image)

    kp1_arr = keyPointsToArray(kp1)
    kp2_arr = keyPointsToArray(kp2)

    M, I, J = splitKeyPoints(H, kp1, kp2)

    keyOriginal = cv2.drawKeypoints(img, M[0][0:7], None, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    keyRotated = cv2.drawKeypoints(warped_image, M[1][0:7], None, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(1, 2, 1)
    plt.imshow(keyOriginal)
    # plt.plot(M[0][0].pt[0], M[0][0].pt[1], 'ro')

    fig.add_subplot(1, 2, 2)
    plt.imshow(keyRotated)
    # plt.plot(M[1][0], M[1][1], 'ro')
    plt.show()

    mk1 = keyPointsToArray(M[0])
    mk2 = keyPointsToArray(M[1])
    M = [mk1, mk2]
    I = keyPointsToArray(I)
    J = keyPointsToArray(J)

    np.savez(params_path + '.npz', H=np.array(H), kp1=np.array(kp1_arr), desc1=np.array(desc1), kp2=np.array(kp2_arr),
             desc2=np.array(desc2), M=np.array(M), I=np.array(I), J=np.array(J))

    return warped_image


def PicturesInFolder(folderPath, resize_path, homography_path, params_path):
    assert (os.path.exists(folderPath))
    for file in os.scandir(folderPath):
        ImagePreProcessing(folderPath + '/' + file.name, resize_path + '/' + file.name,
                           homography_path + '/' + file.name, params_path + '/' + file.name)


def splitKeyPoints(H, kp1, kp2):
    M, I, J = [[], []], [], []
    match_2 = []

    for k1 in kp1:
        kv1 = cv2.perspectiveTransform(np.float32(k1.pt).reshape(-1, 1, 2), H)
        match = False

        for k2 in kp2:
            # vector of (x,y,1)
            kv2 = np.array([k2.pt[0], k2.pt[1]])
            dist = np.linalg.norm(kv1 - kv2)  # L2

            # k1, k2 are match
            if dist <= 1:
                match = True
                match_2.append(k2)
                M[0].append(k1)
                M[1].append(k2)
        # k1 not match to any point in kp2
        if not match:
            I.append(k1)

    J = [item for item in kp2 if item not in match_2]
    return M, I, J


def keyPointsToArray(kp):
    kp_arr = []
    for point in kp:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
                point.class_id)
        kp_arr.append(temp)
    return kp_arr


path = "./temp_photos"
resize_path = './photos'
homography_path = './homography_photos'
params_path = "./params"
PicturesInFolder(path, resize_path, homography_path, params_path)
