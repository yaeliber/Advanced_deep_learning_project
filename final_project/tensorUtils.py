import torch


# param: list of keypoints
# return: tensor of keypoints
def array_to_tensor_of_key_points(arr):
    kp = []
    for k in arr:
        kp.append(torch.Tensor([k[0][0], k[0][1], k[1], k[2], k[3], k[4], k[5]]))
    return torch.stack((kp))


# param: kp1, kp2 are list of keypoints
#        matches are (Dmatch) list of matches between kp1 and kp2
# return: 2 tensors of source and destination from matches
def Dmatch_to_src_dst_tensors(kp1, kp2, matches):
    src = []
    dst = []
    print("line 21 matches", matches)
    for m in matches:
        src.append(torch.Tensor([kp1[m.queryIdx][0], kp1[m.queryIdx][1]]))
        dst.append(torch.Tensor([kp2[m.trainIdx][0], kp2[m.trainIdx][1]]))
    return torch.stack((src)), torch.stack((dst))


# param: kp is list of keypoints
#        tensor_kp is a tensor of keypints (only x and y coordinates)
# return: tensor of keypoints that doesnt exist in tensor_kp
def kp_not_in_tensor(kp, tensor_kp):
    output = []
    for item in kp:
        found = False
        x = item[0]
        y = item[1]
        for point in tensor_kp:
            if point[0] == x and point[1] == y:
                found = True
                break
        if not found:
            output.append(item)
    if len(output) == 0 :
        return torch.empty(0)
    return torch.stack((output))


# return: score of matching (according to M, I, J score)
def get_match_score_tensor(kp1, kp2, best_matches, M, I, J):
    print('--------- In get_match_score ---------')
    # extract keyPoints from params we made on dataSetCreate

    if len(best_matches) == 0:
        print("no matches")
        return 0

    # 2 tensors of source and destination from matches
    src_pts, dst_pts = Dmatch_to_src_dst_tensors(kp1, kp2, best_matches)

    M_ = torch.stack(([src_pts, dst_pts]))
    I_ = kp_not_in_tensor(kp1, src_pts)
    J_ = kp_not_in_tensor(kp2, dst_pts)

    print('len M  ', len(M[0]))
    print('len M* ', len(M_[0]))
    M_counter = 0
    # M_counter is counting intersection between M and M_ (M_ = M*)
    for j in range(len(M_[0])):
        for i in range(len(M[0])):
            if M[0][i][0] == M_[0][j][0] and M[0][i][1] == M_[0][j][1] \
                    and M[1][i][0] == M_[1][j][0] and M[1][i][1] == M_[1][j][1]:
                M_counter += 1
                break

    print('len I  ', len(I))
    print('len I* ', len(I_))
    I_counter = 0
    # I_counter is counting intersection between I and I_ (I_ = I*)
    for kp_1 in I_:
        for kp_2 in I:
            if kp_1[0] == kp_2[0] and kp_1[1] == kp_2[1]:
                I_counter += 1
                break

    print('len J  ', len(J))
    print('len J* ', len(J_))
    J_counter = 0
    # J_counter is counting intersection between J and J_ (J_ = J*)
    for kp_1 in J_:
        for kp_2 in J:
            if kp_1[0] == kp_2[0] and kp_1[1] == kp_2[1]:
                J_counter += 1
                break

    print('-----------------')
    print('M_counter: ', M_counter)
    print('I_counter: ', I_counter)
    print('J_counter: ', J_counter)
    score = (M_counter + I_counter + J_counter) / (len(M[0]) + len(I) + len(J))
    print('match score: ', score)

    return score
