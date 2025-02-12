import torch

def check_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"use device {device} to train the model")
    return device

def main():
    list = []
    for i in range(49):
        list.append(i)
    print(list[29:39])
    print(list[39:])

if __name__ == "__main__": 
    check_device()
