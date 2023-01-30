"""
You will need to validate your NN implementation using PyTorch. You can use any PyTorch functional or modues in this file.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from typing import Optional, List, Tuple, Dict


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class SingleLayerMLP(nn.Module):
    """constructing a single layer neural network with Pytorch"""
    def __init__(self, indim, outdim, hidden_layer=100):
        super(SingleLayerMLP, self).__init__()
        raise NotImplementedError()


    def forward(self, x):
        """
        x shape (batch_size, input_dim)
        """
        raise NotImplementedError()


class DS(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.length = len(X)
        self.X = X
        self.Y = Y

    def __getitem__(self, idx):
        x = self.X[idx, :]
        y = self.Y[idx]
        return (x, y)

    def __len__(self):
        return self.length


def validate(loader):
    """takes in a dataloader, then returns the model loss and accuracy on this loader"""
    raise NotImplementedError()


if __name__ == "__main__":
    """The dataset loaders were provided for you.
    You need to implement your own training process.
    You need plot the loss and accuracies during the training process and test process. 
    """


    indim = 10
    outdim = 2
    hidden_dim = 100
    lr = 0.01
    batch_size = 64
    epochs = 200

    #dataset
    Xtrain = np.loadtxt("/Users/liu5/Documents/ta/2023 spring/HW1/data/XTrain.txt", delimiter="\t")
    Ytrain = np.loadtxt("/Users/liu5/Documents/ta/2023 spring/HW1/data/yTrain.txt", delimiter="\t").astype(int)
    m1, n1 = Xtrain.shape
    print(m1, n1)
    train_ds = DS(Xtrain, Ytrain)
    train_loader = DataLoader(train_ds, batch_size=batch_size)

    Xtest = np.loadtxt("/Users/liu5/Documents/ta/2023 spring/HW1/data/XTest.txt", delimiter="\t")
    Ytest = np.loadtxt("/Users/liu5/Documents/ta/2023 spring/HW1/data/yTest.txt", delimiter="\t").astype(int)
    m2, n2 = Xtest.shape
    print(m1, n2)
    test_ds = DS(Xtest, Ytest)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    #construct the model
    raise NotImplementedError()
    #construct the training process
    raise NotImplementedError()