import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from itertools import combinations, product

from main import *
from CustomDataLoader import *
from tensorUtils import *



def loss_function(match, data, loss_range=1000.0):
    # extract keyPoints from params we made on dataSetCreate
    kp1 = data['kp1']
    kp2 = data['kp2']

    match_score = get_match_score_tensor(kp1, kp2, match, data['M'], data['I'], data['J'])
    match_score = torch.tensor((loss_range - (match_score * loss_range)), requires_grad=True)
    # return loss_range - (match_score * loss_range)
    print("match_score", match_score)
    return match_score

class GAT(torch.nn.Module):
    def __init__(self, in_channels=128, out_channels=128):
        super(GAT, self).__init__()
        self.DB_percentage = torch.nn.Parameter(torch.ones(1) * 0.4, requires_grad=True)
        self.hid = 1
        self.in_head = 128
        self.out_head = 1

        self.conv1 = GATConv(in_channels, self.hid, heads=self.in_head, dropout=0.6)
        self.conv2 = GATConv(self.hid * self.in_head, out_channels, concat=False, heads=self.out_head, dropout=0.6)

    def loss_implement(self, match, p_match, data):
        M = data['M_ind']
        I = data['I_ind']
        J = data['J_ind']
        print("p_match ", p_match)
        p_match -= 0.02
        loss = torch.tensor(0.0, requires_grad=True)
        loss = torch.add(loss, torch.mul(torch.sum(torch.log(p_match[M[0].long(), M[1].long()].exp())), -1))/len(M[0])
        loss = torch.add(loss, torch.mul(torch.sum(torch.log(p_match[I.long(), torch.Tensor([len(data['kp2'])] * len(I)).long()].exp())), -1))/len(I)
        loss = torch.add(loss, torch.mul(torch.sum(torch.log(p_match[torch.Tensor([len(data['kp1'])] * len(J)).long(), J.long()].exp())), -1))/len(J)
        # loss = torch.add(loss, torch.mul(torch.sum(p_match[M[0].long(), M[1].long()]),
        #                                  -1))/len(M[0])   # sum(i∈M[0] and j∈M[1] -log P[i,j])
        # # print("loss1 ", loss)
        # loss = torch.add(loss, torch.mul(
        #     torch.sum(p_match[I.long(), torch.Tensor([len(data['kp2'])] * len(I)).long()]),
        #     -1))/len(I)  # sum(i∈I -log P[i,N+1])
        # # print("loss2 ", loss)
        # loss = torch.add(loss, torch.mul(
        #     torch.sum(p_match[torch.Tensor([len(data['kp1'])] * len(J)).long(), J.long()]),
        #     -1))/len(J)  # sum(j∈J -log P[M+1,j])
        mij_loss = loss_function(match, data)
        print("loss3 ", loss)
        return loss

    # return edge indexes according to descriptors (inside and cross)
    # return edges in shape [[sources], [destinations]]
    def get_edge_index(self, desc1, desc2):
        len1 = len(desc1)
        len2 = len(desc2)

        list1 = list(range(len1))
        list2 = list(range(len1, len1 + len2))

        # All possible pairs inside each group
        inside_temp = list(combinations(list1, 2)) + list(combinations(list2, 2))
        inside_edge = [[], []]
        for s, d in inside_temp:
            inside_edge[0].append(s)
            inside_edge[1].append(d)

        # duplicate inside_edge for direct graph
        temp = inside_edge[0]
        inside_edge[0] = inside_edge[0] + inside_edge[1]
        inside_edge[1] = inside_edge[1] + temp

        # All possible pairs between groups
        cross_edge = [list(np.sort(list1 * len2)), list(list2 * len1)]

        # duplicate cross_edge for direct graph
        temp = cross_edge[0]
        cross_edge[0] = cross_edge[0] + cross_edge[1]
        cross_edge[1] = cross_edge[1] + temp

        return torch.LongTensor(inside_edge),  torch.LongTensor(cross_edge)

    def forward(self, data):
        iters = 1
        desc1, desc2 = data['desc1'], data['desc2']
        # desc1 = F.normalize(desc1, dim = 0)
        # desc2 = F.normalize(desc2, dim = 0)

        inside_edge, cross_edge = self.get_edge_index(desc1, desc2)

        x = torch.Tensor(np.concatenate((desc1, desc2)))
        # for i in range(iters):
        #     print('x shape: ', x.shape)
        #     # print("x before conv1: ", x)
        #     x = self.conv1(x, inside_edge)
        #     # print("x after conv1: ", x)
        #     print('x shape: ', x.shape)
        #     x = F.elu(x)
        # x = self.conv2(x, cross_edge)

        desc1 = x[0:len(desc1)]
        desc2 = x[len(desc1):]
        p_match, match = sinkhorn_match2(desc1, desc2, self.DB_percentage)
        return p_match, match


def train(model, optimizer, loader):
    model.train()

    total_loss = 0
    for data in loader.dataset:
        optimizer.zero_grad()  # Clear gradients.

        p_match, match = model(data)  # Forward pass.
        p_match.retain_grad()
        # match.retain_grad()

        loss = model.loss_implement(match, p_match, data)  # Loss computation.
        # print("params before: ")
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         # param.retain_grad() #??
        #         print(name, param.grad)
        # loss.retain_grad()
        # print("loss.grad", loss.grad)
        loss.backward()  # Backward pass.
        # print("loss.grad", loss.grad)
        optimizer.step()  # Update model parameters.
        # print("params after: ")
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.grad)
        total_loss += loss.item()

    return total_loss / len(loader.dataset)


@torch.no_grad()
def test(model, loader):
    model.eval()

    total_correct = 0
    for data in loader.dataset:
        match = model(data)
        # pred = logits.argmax(dim=-1)
        # total_correct += int((pred == data.y).sum())

    return total_correct / len(loader.dataset)


if __name__ == '__main__':
    train_csv_path = '../../data/params/files_train_name.csv'
    test_csv_path = '../../data/params/files_test_name.csv'
    npz_folder_path = '../../data/params/' + '2'
    train_dataset = NpzDataLoader(train_csv_path, npz_folder_path)
    # test_dataset = NpzDataLoader(test_csv_path, npz_folder_path)

    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=False)  # num_workers??
    # test_loader = DataLoader(test_dataset, batch_size=20)  # num_workers?? batch_size??

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cpu"

    model = GAT().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    for epoch in range(1, 51):
        loss = train(model, optimizer, train_loader)
        # test_acc = test(model, test_loader)
        # print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Test Accuracy: {test_acc:.4f}')
