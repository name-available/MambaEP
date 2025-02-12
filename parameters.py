import argparse


def get_args():
    parser = argparse.ArgumentParser('Training and evaluation parameters', add_help=False)

    # load parameters
    parser.add_argument('--if_pretrained', type=bool, default=False, help='If the model is pre trained.')

    # input parameters
    parser.add_argument('--data_path', type=str, default='E:\dataset', help='Path of dataset.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--val_batch_size', type=int, default=8, help='Batch size for validation.')

    # model parameters
    parser.add_argument('--sub', type=int, default=1, help='Subsampling factor.')
    parser.add_argument('--T_in', type=int, default=10, help='Number of input time steps.')
    parser.add_argument('--T_out', type=int, default=10, help='Number of output time steps.')
    parser.add_argument('--reshape', type=int, nargs='+', help='Optional reshape permutation.')

    # training parameters
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs.')

    # logging parameters
    parser.add_argument('--log_per_samples', type=int, default=10, help='Evaluate every n epochs.')
    parser.add_argument('--log_path', type=str, default="logs/maep_moving_mnist.log", help='The path of logging file.')
    parser.add_argument('--checkpoints', type=str, default="chackpoints/maep_model.pth", help='Trained model checkpoints.')
    return parser.parse_args()

