import matplotlib.pyplot as plt
import numpy as np
import cv2

# np.save('./data', np.array([1, 2]))

# H = np.array([[1, 0, 1], [0, 0, 0], [0, 9, 0]])
# # img = np.array([[1, 1, 1]])
# np.savez('./data.npz', H=H)
# print(np.load('./data.npz')['H'])
# print("--------------------------")
# img = np.array([[1, 1, 1]])
# np.savez('./data.npz', img=img)

img = cv2.cvtColor(cv2.imread('../../data/error_img/paris_museedorsay_000904.jpg'), cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
feature_extractor = cv2.SIFT_create()
kp1, desc1 = feature_extractor.detectAndCompute(img, None)
print("img1", img.shape)
print("kp1: ", kp1)
print("len kp1: ", len(kp1))




