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
    print("====================================================================")
    for k in kp1:
        print("x = ", k.pt[0], "     y = ", k.pt[1])
    print("====================================================================")
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
    kp1_arr = []
    kp2_arr = []
    for point in kp1:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
                point.class_id)
        kp1_arr.append(temp)

    for point in kp2:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
                point.class_id)
        kp2_arr.append(temp)

    np.savez(params_path + '.npz', H=np.array(H), kp1=np.array(kp1_arr), desc1=np.array(desc1), kp2=np.array(kp2_arr),
             desc2=np.array(desc2))

    return warped_image


def PicturesInFolder(folderPath, resize_path, homography_path, params_path):
    assert (os.path.exists(folderPath))
    for file in os.scandir(folderPath):
        ImagePreProcessing(folderPath + '/' + file.name, resize_path + '/' + file.name,
                           homography_path + '/' + file.name, params_path + '/' + file.name)


def readNpzFiles(path):
    assert (os.path.exists(path))
    for file in os.scandir(path):
        data = np.load(path + '/' + file.name, allow_pickle = True)
        print("H: ", data['H'])
        print("desc1: ", data['desc1'])
        print("kp1: ", data['kp1'])
        print()


path = "./temp_photos"
resize_path = './photos'
homography_path = './homography_photos'
params_path = "./params"
# PicturesInFolder(path, resize_path, homography_path, params_path)
readNpzFiles(params_path)

# path = '../../datasets/paris_1/paris/eiffel'
# resize_path = '../../resize_datasets/eiffel'
# homography_path = '../../homography_datasets/eiffel'
# d.PicturesInFolder(path, resize_path, homography_path)
