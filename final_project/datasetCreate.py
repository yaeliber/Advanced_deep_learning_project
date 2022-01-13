import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt


def get_keypoints_and_descriptors(img1, img2):
    print('--------- In get_keypoints_and_descriptors ---------')
    # use orb if sift is not installed
    feature_extractor = cv2.SIFT_create()

    # find the keypoints and descriptors with chosen feature_extractor
    kp1, desc1 = feature_extractor.detectAndCompute(img1, None)
    kp2, desc2 = feature_extractor.detectAndCompute(img2, None)

    print('img1', img1.shape)
    print('kp1', len(kp1))
    print('kp2', len(kp2))

    keyOriginal = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    keyRotated = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # fig = plt.figure(figsize=(10, 10))
    # fig.add_subplot(1, 2, 1)
    # plt.title('keyOriginalPoints')
    # plt.axis('off')
    # plt.imshow(keyOriginal)
    #
    # fig.add_subplot(1, 2, 2)
    # plt.title('keyRotatedPoints')
    # plt.axis('off')
    # plt.imshow(keyRotated)
    # plt.show()
    print('\n\n')

    return kp1, desc1, kp2, desc2


def get_kp_distance(kp_arr, ind1, ind2):
    x1, y1 = kp_arr[ind1][0][0], kp_arr[ind1][0][1]
    x2, y2 = kp_arr[ind2][0][0], kp_arr[ind2][0][1]

    distance = np.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))
    return distance


def delete_close_key_points(kp_arr, desc):
    radius = 5
    k1 = 0
    length = len(kp_arr)
    while k1 < length:

        k2 = k1 + 1
        while k2 < length:
            # delete all the close key points and descriptors that are close to k1
            if get_kp_distance(kp_arr, k1, k2) < radius:
                kp_arr = np.delete(kp_arr, k2, axis=0)
                desc = np.delete(desc, k2, axis=0)
                k2 -= 1

            length = len(kp_arr)
            k2 += 1

        length = len(kp_arr)
        k1 += 1

    return kp_arr, desc


def image_pre_processing(img_name, path, resize_Path, homography_path, params_path):
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

    kp1, desc1, kp2, desc2 = get_keypoints_and_descriptors(img, warped_image)

    kp1_arr = key_points_to_array(kp1)
    kp2_arr = key_points_to_array(kp2)

    kp1_arr, desc1 = delete_close_key_points(kp1_arr, desc1)
    kp2_arr, desc2 = delete_close_key_points(kp2_arr, desc2)

    if len(kp1_arr) < 4 or len(kp2_arr) < 4:
        print('remove')
        cv2.imwrite('../../data/error_img/' + img_name, cv2.imread(path))
        os.remove(path)
        return

    cv2.imwrite(resize_Path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(homography_path, cv2.cvtColor(warped_image, cv2.COLOR_RGB2BGR))

    M, I, J = split_key_points(H, kp1, kp2)

    keyOriginal = cv2.drawKeypoints(img, M[0][0:7], None, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    keyRotated = cv2.drawKeypoints(warped_image, M[1][0:7], None, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    # fig = plt.figure(figsize=(10, 10))
    # fig.add_subplot(1, 2, 1)
    # plt.imshow(keyOriginal)
    # plt.plot(M[0][0].pt[0], M[0][0].pt[1], 'ro')

    # fig.add_subplot(1, 2, 2)
    # plt.imshow(keyRotated)
    # plt.plot(M[1][0], M[1][1], 'ro')
    # plt.show()

    mk1 = key_points_to_array(M[0])
    mk2 = key_points_to_array(M[1])
    M = [mk1, mk2]
    I = key_points_to_array(I)
    J = key_points_to_array(J)

    H_mean, H_std = get_difficult_level(H)

    np.savez(params_path + '.npz', H=np.array(H), kp1=np.array(kp1_arr), desc1=np.array(desc1), kp2=np.array(kp2_arr),
             desc2=np.array(desc2), M=np.array(M), I=np.array(I), J=np.array(J), H_mean=H_mean, H_std=H_std)

    return warped_image


def pictures_in_folder(folderPath, resize_path, homography_path, params_path):
    assert (os.path.exists(folderPath))
    for file in os.scandir(folderPath):
        print(file.name)
        image_pre_processing(file.name, folderPath + '/' + file.name, resize_path + '/' + file.name,
                             homography_path + '/' + file.name, params_path + '/' + file.name)


def split_key_points(H, kp1, kp2):
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


def key_points_to_array(kp):
    kp_arr = []
    for point in kp:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
                point.class_id)
        kp_arr.append(temp)
    return kp_arr


def get_difficult_level(H):
    # if the H is close to I we are in a simple case (the new image close to the original)
    # |H-I|.mean => 0
    # |H-I|.std => 1
    I = np.eye(3)
    dif = np.abs(H - I)
    H_mean = np.mean(dif)
    H_std = np.std(dif)
    return H_mean, H_std


# path = './data/temp_photos'
# resize_path = './data/resize_photos'
# homography_path = './data/homography_photos/1'
# params_path = './data/params/1'
# PicturesInFolder(path, resize_path, homography_path, params_path)

path = '../../data/restart_img'
resize_path = '../../data/resize_photos'
homography_path = '../../data/homography_photos/1'
params_path = '../../data/params/1'
pictures_in_folder(path, resize_path, homography_path, params_path)

# homography_path = './data/homography_photos/2'
# params_path = './data/params/2'
# PicturesInFolder(path, resize_path, homography_path, params_path)
