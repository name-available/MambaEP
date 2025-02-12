import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from dataloader.dataloader_moving_mnist import load_data as load_moving_mnist
from MambaEP import MambaEarthPred_model
from tools.check import check_device
from tools.train_function import train_model, test_model
from parameters import get_args

def main(args):
    train_loader, eval_loader, test_loader, _, _ = load_moving_mnist(
        batch_size = args.batch_size,
        val_batch_size = args.batch_size
    )

    model = MambaEarthPred_model(shape_in=(10, 1, 64, 64))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    device = check_device()

    model.to(device)


    train_model(model, train_loader, eval_loader, criterion, optimizer, device, args)

    model.load_state_dict(torch.load(args.checkpoints))

    test_loss = test_model(model, test_loader, device)
    print(f'TEST:::MSE Loss: {test_loss:.7f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Training and evaluation parameters', add_help=False)

    # input parameters
    parser.add_argument('--data_path', type=str, default='/userhome/cs2/wang1210/dataset', help='Path of dataset.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training.')
    parser.add_argument('--val_batch_size', type=int, default=4, help='Batch size for validation.')

    # model parameters
    parser.add_argument('--sub', type=int, default=1, help='Subsampling factor.')
    parser.add_argument('--T_in', type=int, default=10, help='Number of input time steps.')
    parser.add_argument('--T_out', type=int, default=10, help='Number of output time steps.')
    parser.add_argument('--reshape', type=int, nargs='+', help='Optional reshape permutation.')

    # training parameters
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--if_pretrained', type=bool, default=False, help='If have pre trained model pth.')

    # logging parameters
    parser.add_argument('--log_per_samples', type=int, default=500, help='Evaluate every n epochs.')
    parser.add_argument('--log_path', type=str, default="logs/ablation_full.log", help='The path of logging file.')
    parser.add_argument('--checkpoints', type=str, default="checkpoints/mamep_mm.pth", help='Trained model checkpoints.')

    args = parser.parse_args()
    main(args)
