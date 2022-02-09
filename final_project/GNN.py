from main import *


def loss(data, DB_P, loss_range=1000):
    match = sinkhorn_match(data['desc1'], data['desc2'], DB_P)

    # extract keyPoints from params we made on dataSetCreate
    kp1 = array_to_key_points(data['kp1'])
    kp2 = array_to_key_points(data['kp2'])

    match_score = get_match_score(kp1, kp2, match, data['M'], data['I'], data['J'])

    return loss_range - (match_score * loss_range)

path = "../../data/params/delete_close_kp/paris_defense_000038.jpg.npz"
data = np.load(path, allow_pickle=True)
arr_loss = []
for i in range(1, 5):
    res_loss = loss(data, 0.2*i)
    print("loss: ",res_loss, "DB: ", 0.2*i)
    arr_loss.append((0.2*i,res_loss ))
print(arr_loss)