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


def loss(match, data, loss_range=1000):
    # extract keyPoints from params we made on dataSetCreate
    kp1 = array_to_key_points(data['kp1'])
    kp2 = array_to_key_points(data['kp2'])

    match_score = get_match_score(kp1, kp2, match, data['M'], data['I'], data['J'])

    return loss_range - (match_score * loss_range)


class GAT(torch.nn.Module):
    def __init__(self, in_channels=128, out_channels=128):
        super(GAT, self).__init__()
        self.DB_percentage = Variable(torch.tensor(0.4), requires_grad=True)
        self.hid = 1
        self.in_head = 8
        self.out_head = 1

        self.conv1 = GATConv(in_channels, self.hid, heads=self.in_head, dropout=0.6)
        self.conv2 = GATConv(self.hid * self.in_head, out_channels, concat=False, heads=self.out_head, dropout=0.6)

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

        return inside_edge, cross_edge

    def forward(self, data):
        iters = 4
        desc1, desc2 = data['desc1'], data['desc2']
        inside_edge, cross_edge = self.get_edge_index(desc1, desc2)

        x = desc1 + desc2
        for i in range(iters):
            x = self.conv1(x, inside_edge)
            x = F.elu(x)
            x = self.conv1(x, cross_edge)
            x = F.elu(x)

        x = self.conv2(x, cross_edge)

        desc1 = x[0:len(desc1)]
        desc2 = x[len(desc1):]
        match = sinkhorn_match(desc1, desc2, self.DB_percentage.item())
        return match


def train(model, optimizer, loader):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()  # Clear gradients.
        match = model(data)  # Forward pass.
        loss = loss(match, data)  # Loss computation.
        loss.backward()  # Backward pass.
        optimizer.step()  # Update model parameters.
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)

if __name__ == '__main__':
    csv_path = '../../data/params/files_name.csv'
    npz_folder_path = '../../data/params/' + 1
    dl = NpzDataLoader(csv_path, npz_folder_path)
    train_loader = DataLoader(dl, batch_size=20, shuffle=False)  # num_workers??

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cpu"

    model = GAT().to(device)
    data = train_loader[0].to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    model.train()
    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

        if epoch % 200 == 0:
            print(loss)

        loss.backward()
        optimizer.step()