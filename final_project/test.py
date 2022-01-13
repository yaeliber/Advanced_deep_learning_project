import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# np.save('./data', np.array([1, 2]))

# H = np.array([[1, 0, 1], [0, 0, 0], [0, 9, 0]])
# # img = np.array([[1, 1, 1]])
# np.savez('./data.npz', H=H)
# print(np.load('./data.npz')['H'])
# print("--------------------------")
# img = np.array([[1, 1, 1]])
# np.savez('./data.npz', img=img)

# img = cv2.cvtColor(cv2.imread('../../data/error_img/paris_museedorsay_000904.jpg'), cv2.COLOR_BGR2RGB)
# plt.imshow(img)
# plt.show()
# feature_extractor = cv2.SIFT_create()
# kp1, desc1 = feature_extractor.detectAndCompute(img, None)
# print("img1", img.shape)
# print("kp1: ", kp1)
# print("len kp1: ", len(kp1))

# count = 0
# for file in os.scandir('../../data/params/1/'):
#     data = np.load(file, allow_pickle=True)#path3 npz
#     H_mean = data['H_mean']
#
#     if H_mean > 200:#1.25*1e6:
#         print(file.name, "   ",H_mean )
#         count += 1
#
# print("count: ", count)

def array_to_key_points(arr):
    kp = []
    for k in arr:
        kp.append(cv2.KeyPoint(k[0][0], k[0][1], k[1], k[2], k[3], k[4], k[5]))
    return kp

def get_kp_distance(kp_arr, ind1, ind2):
    print(kp_arr.shape)
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

img = cv2.cvtColor(cv2.imread('../../data/restart_img/paris_general_002907.jpg'), cv2.COLOR_BGR2RGB)# path after resize
data = np.load('../../data/params/1/paris_general_002907.jpg.npz', allow_pickle=True)# path npz
kp1 = array_to_key_points(data['kp1'])
desc1 = data['desc1']
print(len(kp1), " of key point before delete")
key_original = cv2.drawKeypoints(img, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

kp1 = data['kp1']
kp1_arr, desc1 = delete_close_key_points(kp1, desc1)
kp1_arr = array_to_key_points(kp1_arr)
print(len(kp1_arr), " of key point after delete close key points")

after_delete_close_kp = cv2.drawKeypoints(img, kp1_arr, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
fig = plt.figure(figsize=(10, 10))
fig.add_subplot(1, 2, 1)
plt.imshow(key_original)

fig.add_subplot(1, 2, 2)
plt.imshow(after_delete_close_kp)
plt.show()






