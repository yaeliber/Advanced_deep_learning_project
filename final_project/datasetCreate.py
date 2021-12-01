import os
import cv2
import random
import numpy as np
from numpy.linalg import inv


class dataset:
    def __init__(self):
        pass
        # self.folderPath = folderPath

    def ImagePreProcessing(self, path, resize_Path, homography_path):
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

        H = cv2.getPerspectiveTransform(np.float32(
            four_points), np.float32(perturbed_four_points))
        H_inverse = inv(H)

        warped_image = cv2.warpPerspective(img, H_inverse, (320, 240))

        cv2.imwrite(resize_Path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(homography_path, cv2.cvtColor(
            warped_image, cv2.COLOR_RGB2BGR))
        return warped_image

    def PicturesInFolder(self, folderPath, resize_path, homography_path):
        assert (os.path.exists(folderPath))
        for file in os.scandir(folderPath):
            self.ImagePreProcessing(folderPath + '/' + file.name, resize_path + '/' + file.name,
                                    homography_path + '/' + file.name)


d = dataset()
path = "./temp_photos"
resize_path = './photos'
homography_path = './homography_photos'
d.PicturesInFolder(path, resize_path, homography_path)

# path = '../../datasets/paris_1/paris/eiffel'
# resize_path = '../../resize_datasets/eiffel'
# homography_path = '../../homography_datasets/eiffel'
# d.PicturesInFolder(path, resize_path, homography_path)
