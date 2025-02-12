import torch

def check_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"use device {device} to train the model")
    return device

if __name__ == "__main__": 
    check_device()
