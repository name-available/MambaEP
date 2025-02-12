import numpy as np
import scipy
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class NS2DDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = torch.from_numpy(np.load(data_path))
        self.data.unsqueeze_(2)  
        self.transform = transform
        self.mean = 0
        self.std = 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_frames = self.data[idx][:10]
        output_frames = self.data[idx][10:]
        return input_frames, output_frames


def load_data_mat(path, sub=1, T_in=10, T_out=10, batch_size=64, reshape=None):
    ntrain = 1000
    neval = 100
    ntest = 100
    total = ntrain + neval + ntest
    f = scipy.io.loadmat(path)
    data = f['u'][..., 0:total]
    data = torch.tensor(data, dtype=torch.float32)

    train_a = data[:ntrain, ::sub, ::sub, :T_in]
    train_u = data[:ntrain, ::sub, ::sub, T_in:T_out+T_in]
    train_a = train_a.unsqueeze(-1).permute(0, 3, 1, 2, 4).permute(0, 1, 4, 2, 3)
    train_u = train_u.unsqueeze(-1).permute(0, 3, 1, 2, 4).permute(0, 1, 4, 2, 3)

    eval_a = data[ntrain:ntrain + neval, ::sub, ::sub, :T_in]
    eval_u = data[ntrain:ntrain + neval, ::sub, ::sub, T_in:T_out+T_in]
    eval_a = eval_a.unsqueeze(-1).permute(0, 3, 1, 2, 4).permute(0, 1, 4, 2, 3)
    eval_u = eval_u.unsqueeze(-1).permute(0, 3, 1, 2, 4).permute(0, 1, 4, 2, 3)

    test_a = data[ntrain + neval:ntrain + neval + ntest, ::sub, ::sub, :T_in]
    test_u = data[ntrain + neval:ntrain + neval + ntest, ::sub, ::sub, T_in:T_out+T_in]
    test_a = test_a.unsqueeze(-1).permute(0, 3, 1, 2, 4).permute(0, 1, 4, 2, 3)
    test_u = test_u.unsqueeze(-1).permute(0, 3, 1, 2, 4).permute(0, 1, 4, 2, 3)

    if reshape:
        train_a = train_a.permute(reshape)
        train_u = train_u.permute(reshape)
        eval_a = eval_a.permute(reshape)
        eval_u = eval_u.permute(reshape)
        test_a = test_a.permute(reshape)
        test_u = test_u.permute(reshape)

    train_loader = DataLoader(TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(TensorDataset(eval_a, eval_u), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

    return train_loader, eval_loader, test_loader


def load_data(batch_size, val_batch_size, data_root, num_workers):
    train_dataset = NS2DDataset(data_path=data_root + 'ns_V1e-4_train.npy', transform=None)
    test_dataset = NS2DDataset(data_path=data_root + 'ns_V1e-4_test.npy', transform=None)
    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                  num_workers=num_workers)
    dataloader_validation = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, pin_memory=True,
                                       num_workers=num_workers)
    dataloader_test = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, pin_memory=True,
                                 num_workers=num_workers)

    mean, std = 0, 1
    return dataloader_train, dataloader_validation, dataloader_test, mean, std

