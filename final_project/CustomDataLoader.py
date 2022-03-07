import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class NpzDataLoader(Dataset):
    """NPZ dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the npz files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.files_names = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.files_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        npz_file_name = os.path.join(self.root_dir, self.files_names.iloc[idx, 0])
        data = np.load(npz_file_name, allow_pickle=True)
        data = dict(data)
        data['name'] = self.files_names.iloc[idx, 0][:-4]

        if self.transform:
            data = self.transform(data)

        return data

# csv_file = '../../data/params/files_name.csv'
# root_dir = '../../data/params/1/'
# img_dataset = NpzDataLoader(csv_file = csv_file, root_dir = root_dir)
#
# for i in range(len(img_dataset)):
#     sample = img_dataset[i]
#     print(i)
#     print('name', sample['name'])
#     # sample['H_std'], sample['desc2'])

