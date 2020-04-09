
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.transforms as tf

from lienp.dataset import RotationMNIST


def train(model, dataloader, criterion, optimizer):
    pass


def test(model, dataloader, criterion):
    pass


def main():
    trainset = RotationMNIST("~/data/rotmnist", train=True)
    testset = RotationMNIST("~/data/rotmnist", train=False)

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    model = None

    creterion = None
    optimizer = optim.Adam(model.paramters(), lr=1e-4)

    for epoch in range(epochs):
        train(model, trainloader, criterion, optimizer)
        test(model, testloader, criterion)





if __name__ == '__main__':
    main()