import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt


class VilDataset(Dataset):
    def __init__(self, data, transform=None):
        # self.data = data
        self.data = data[:, :40]

        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        img = self.data[index].reshape(40, 1, 192, 192)
        # img = self.data[index].reshape(20, 1, 192, 192)
        
        if self.transform:
            img = self.transform(img)

        input_img = img[:20]
        output_img = img[20:]

        input_img = torch.from_numpy(input_img).contiguous().float()
        output_img = torch.from_numpy(output_img).contiguous().float()
        return input_img, output_img


def load_data(batch_size, num_workers = 0, 
              data_root = "/userhome/cs2/wang1210/dataset/Sevir"):
    data = np.load(f'{data_root}/SEVIR_ir069.npy', mmap_mode='r')

    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=2 / 3, random_state=42)  # 2/3 of 30% is 20%

    train_set = VilDataset(train_data)
    val_set = VilDataset(val_data)
    test_set = VilDataset(test_data)

    dataloader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True,
                                  num_workers=num_workers)
    dataloader_validation = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True,
                                       num_workers=num_workers)
    dataloader_test = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True,
                                 num_workers=num_workers)

    mean, std = 0, 1

    return dataloader_train, dataloader_validation, dataloader_test, mean, std


def img_show():
    batch_size = 1
    num_workers = 0
    data_root = "/userhome/cs2/wang1210/dataset/Sevir"

    dataloader_train, dataloader_validation, dataloader_test, mean, std = load_data(
        batch_size=batch_size, data_root=data_root, num_workers=num_workers
    )

    data_iter = iter(dataloader_test)
    input_img, output_img = next(data_iter)

    input_img = input_img.numpy()
    output_img = output_img.numpy()

    for i in range(20):
        fig, ax = plt.subplots()
        ax.imshow(input_img[0, i, 0, :, :], cmap='RdBu')
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(f'image/sevir_input_images_{i}.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"sevir_{i} input image saved")

    for i in range(20):
        fig, ax = plt.subplots()
        ax.imshow(output_img[0, i, 0, :, :], cmap='RdBu')
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(f'image/sevir_target_images_{i}.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"sevir_{i} target image saved")

def check_shape():
    batch_size = 1
    num_workers = 0
    data_root = "/userhome/cs2/wang1210/dataset/Sevir"

    dataloader_train, dataloader_validation, dataloader_test, mean, std = load_data(
        batch_size = batch_size, data_root = data_root, num_workers = num_workers
    )
    print(dataloader_train.dataset.data.shape)
    print(dataloader_validation.dataset.data.shape)
    print(dataloader_test.dataset.data.shape)



if __name__ == '__main__':
    img_show()