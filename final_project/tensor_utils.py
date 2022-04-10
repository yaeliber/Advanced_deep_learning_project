import numpy as np
import cv2
import torch

def array_to_tensor_of_key_points(arr):
    kp = []
    for k in arr:
        kp.append(torch.Tensor([k[0][0], k[0][1], k[1], k[2], k[3], k[4], k[5]]))
    return torch.stack((kp))

def get_match_score_tensor(kp1, kp2, best_matches, M, I, J):
    print('--------- In get_match_score ---------')
    # extract keyPoints from params we made on dataSetCreate

    src_pts = np.float32([kp1[m.queryIdx].pt for m in best_matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in best_matches])

    M_ = [src_pts, dst_pts]
    I_ = [item for item in kp1 if item.pt not in src_pts]
    J_ = [item for item in kp2 if item.pt not in dst_pts]

    M_counter = 0
    print('len M  ', len(M[0]))
    print('len M* ', len(M_[0]))
    for j in range(len(M_[0])):
        for i in range(len(M[0])):
            if M[0][i].pt[0] == M_[0][j][0] and M[0][i].pt[1] == M_[0][j][1] \
                    and M[1][i].pt[0] == M_[1][j][0] and M[1][i].pt[1] == M_[1][j][1]:
                M_counter += 1
                break
    I = array_to_key_points(I)
    print('len I  ', len(I))
    print('len I* ', len(I_))
    I_counter = 0
    for kp_1 in I_:
        for kp_2 in I:
            if kp_1.pt[0] == kp_2.pt[0] and kp_1.pt[1] == kp_2.pt[1]:
                I_counter += 1
                break

    J = array_to_key_points(J)
    print('len J  ', len(J))
    print('len J* ', len(J_))
    J_counter = 0
    for kp_1 in J_:
        for kp_2 in J:
            if kp_1.pt[0] == kp_2.pt[0] and kp_1.pt[1] == kp_2.pt[1]:
                J_counter += 1
                break

    print('-----------------')
    print('M_counter: ', M_counter)
    print('I_counter: ', I_counter)
    print('J_counter: ', J_counter)
    score = (M_counter + I_counter + J_counter) / (len(M[0]) + len(I) + len(J))
    print('match score: ', score)

    return score