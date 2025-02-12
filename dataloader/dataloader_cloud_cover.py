import pickle
import os
import torch
import numpy as np
from torch.utils.data import Dataset


class CloudCloveDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return self.data['x'].shape[0]

    def __getitem__(self, index):
        input_img = self.data["x"][index].reshape(12, 1, 32, 64) 
        output_img = self.data["y"][index].reshape(12, 1, 32, 64)
        if self.transform:
            img = self.transform(img)

        input_img = torch.from_numpy(input_img).contiguous().float()
        output_img = torch.from_numpy(output_img).contiguous().float()
        return input_img, output_img
    
class MixedCloudCloveDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return self.data['x'].shape[0]

    def __getitem__(self, index):
        input_mix = np.concatenate((self.data["x"][index], self.data["context"][index]), axis=2)
        input_img = input_mix.reshape(12, 9, 32, 64) 
        output_img = self.data["y"][index].reshape(12, 1, 32, 64)
        if self.transform:
            img = self.transform(img)

        input_img = torch.from_numpy(input_img).contiguous().float()
        output_img = torch.from_numpy(output_img).contiguous().float()
        return input_img, output_img


def load_data(mixed = False, root = '/userhome/cs2/wang1210/dataset/cloud_cover/cloud_cover', batch_size = 8):
    train_path = os.path.join(root, 'trn.pkl')
    test_path = os.path.join(root, 'test.pkl')
    val_path = os.path.join(root, 'val.pkl')
    
    with open(train_path, 'rb') as file:
        train_set = pickle.load(file)
    with open(test_path, 'rb') as file:
        test_set = pickle.load(file)
    with open(val_path, 'rb') as file:
        val_set = pickle.load(file)

    if mixed:
        train_set = MixedCloudCloveDataset(train_set)
        test_set = MixedCloudCloveDataset(test_set)
        val_set = MixedCloudCloveDataset(val_set)
    else:
        train_set = CloudCloveDataset(train_set)
        test_set = CloudCloveDataset(test_set)
        val_set = CloudCloveDataset(val_set)

    train_set = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    val_set = torch.utils.data.DataLoader(
    val_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    test_set = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    
    mean, std = 0, 1
    return train_set, val_set, test_set, mean, std

def load_position_info(root = '/userhome/cs2/wang1210/dataset/cloud_cover/cloud_cover'):
    position_path = os.path.join(root, 'position.pkl')
    with open(position_path, 'rb') as file:
        position_info = pickle.load(file)
    return position_info

def main():
    train_loader, val_loader, test_loader,_, _ = load_data(mixed=True, batch_size=1)
    print(len(train_loader))
    print(len(val_loader))
    print(len(test_loader))

if __name__ == '__main__':
    main()
