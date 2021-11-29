import os
import cv2
import random
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

class dataset:
    def __init__(self):
        pass
        # self.folderPath = folderPath

    def ImagePreProcessing(self, path, newPath):
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

        H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
        H_inverse = inv(H)

        warped_image = cv2.warpPerspective(img, H_inverse, (320, 240))
        # plt.title("warped_image")
        # plt.axis("off")
        # plt.imshow(warped_image)
        # plt.show()
        annotated_warp_image = warped_image.copy()
        # create path to the warped image
        # tempArr = path.split(".")
        # newPath = "."
        # for i in range(len(tempArr)):
        #     if i == (len(tempArr) - 1):
        #         newPath += "new."
        #     newPath += tempArr[i]
        newPath = "." + newPath
        print(newPath)
        # './temp_photos/warped_image.jpg'
        cv2.imwrite(newPath, cv2.cvtColor(warped_image, cv2.COLOR_RGB2BGR))
        return warped_image

    def PicturesInFolder(self, folderPath):
        for file in os.scandir(folderPath):
            ImagePreProcessing(folderPath + '/' + file.name, folderPath + '/new' + file.name)

d = dataset()
d.PicturesInFolder("./temp_photos")