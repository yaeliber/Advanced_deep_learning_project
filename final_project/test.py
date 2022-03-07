import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import pandas as pd

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
import pandas as pd


def array_to_key_points(arr):
    kp = []
    for k in arr:
        kp.append(cv2.KeyPoint(k[0][0], k[0][1], k[1], k[2], k[3], k[4], k[5]))
    return kp


def key_points_to_array(kp):
    kp_arr = []
    for point in kp:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
                point.class_id)
        kp_arr.append(temp)
    return kp_arr


def get_kp_distance(kp_arr, ind1, ind2):
    # print(kp_arr.shape)
    x1, y1 = kp_arr[ind1][0][0], kp_arr[ind1][0][1]
    x2, y2 = kp_arr[ind2][0][0], kp_arr[ind2][0][1]

    distance = np.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))
    return distance


def delete_close_key_points(kp_arr, desc):
    radius = 0.2
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


def image_pre_processing(img_name, path, old_params_path, new_params_path):
    data = np.load(old_params_path, allow_pickle=True)
    H = data['H']
    kp1_arr = data['kp1']
    desc1 = data['desc1']
    kp2_arr = data['kp2']
    desc2 = data['desc2']

    kp1_arr, desc1 = delete_close_key_points(kp1_arr, desc1)
    kp2_arr, desc2 = delete_close_key_points(kp2_arr, desc2)

    if len(kp1_arr) < 4 or len(kp2_arr) < 4:
        print('remove')
        cv2.imwrite('../../data/error2_img/' + img_name, cv2.imread(path))
        os.remove(path)
        return

    kp1 = array_to_key_points(kp1_arr)
    kp2 = array_to_key_points(kp2_arr)
    M, I, J = split_key_points(H, kp1, kp2)

    mk1 = key_points_to_array(M[0])
    mk2 = key_points_to_array(M[1])
    M = [mk1, mk2]
    I = key_points_to_array(I)
    J = key_points_to_array(J)

    H_mean, H_std = data['H_mean'], data['H_std']

    np.savez(new_params_path + '.npz', H=np.array(H), kp1=np.array(kp1_arr), desc1=np.array(desc1),
             kp2=np.array(kp2_arr),
             desc2=np.array(desc2), M=np.array(M), I=np.array(I), J=np.array(J), H_mean=H_mean, H_std=H_std)


def pictures_in_folder(folderPath, old_params_path, new_params_path):
    assert (os.path.exists(folderPath))
    for file in os.scandir(folderPath):
        print(file.name)
        if (file.name != 'desktop.ini'):
            image_pre_processing(file.name, folderPath + '/' + file.name, old_params_path + '/' + file.name + '.npz',
                                 new_params_path + '/' + file.name)


def create_pandas_file(folderPath, results_path):
    files_name = []
    assert (os.path.exists(folderPath))
    for file in os.scandir(folderPath):
        if (file.name != 'desktop.ini'):
            files_name.append(file.name + '.npz')
    df = pd.DataFrame(zip(files_name), columns=['name'])
    df.to_csv(results_path, index=False)
    print(len(files_name))


path = '../../data/original_photos'
results_path = '../../data/params/files_name.csv'
# create_pandas_file(path, results_path)
df = pd.read_csv(results_path)
print(df)
print(df.iloc[0, 0])# second 0 for 'name'
# old_params_path = '../../data/params/1'
# new_params_path = '../../data/params/delete_close_kp'
# pictures_in_folder(path, old_params_path, new_params_path)

# img = cv2.cvtColor(cv2.imread('../../data/restart_img/paris_general_002907.jpg'), cv2.COLOR_BGR2RGB)# path after resize
# data = np.load('../../data/params/1/paris_general_002907.jpg.npz', allow_pickle=True)# path npz
# kp1 = array_to_key_points(data['kp1'])
# desc1 = data['desc1']
# print(len(kp1), " of key point before delete")
# key_original = cv2.drawKeypoints(img, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
# kp1 = data['kp1']
# kp1_arr, desc1 = delete_close_key_points(kp1, desc1)
# kp1_arr = array_to_key_points(kp1_arr)
# print(len(kp1_arr), " of key point after delete close key points")
#
# after_delete_close_kp = cv2.drawKeypoints(img, kp1_arr, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# fig = plt.figure(figsize=(10, 10))
# fig.add_subplot(1, 2, 1)
# plt.title('key_original')
# plt.imshow(key_original)
#
# fig.add_subplot(1, 2, 2)
# plt.title('after_delete_close_kp')
# plt.imshow(after_delete_close_kp)
# plt.show()
