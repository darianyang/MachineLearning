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
        self.linear1 = nn.Linear(indim, hidden_layer)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_layer, outdim)


    def forward(self, x):
        """
        x shape (indim, batch_size)
        """
        x1 = self.linear1.forward(x)
        x2 = self.relu.forward(x1)
        x3 = self.linear2.forward(x2)
        return x3


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
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    Ys, Y_hats = [], []
    loss = 0
    for xs, ys in loader:
        xs = xs.to(device).double()
        ys = ys.to(device).to(torch.long)
        out = model(xs).detach()
        loss += criterion(out, ys)
        yhat = np.argmax(out.to("cpu").numpy(), axis=1)
        
        Ys.append(ys.detach().to("cpu"))
        Y_hats.append(yhat)
    Ys = np.concatenate(Ys, axis=0).reshape(-1,)
    Y_hats = np.concatenate(Y_hats, axis=0).reshape(-1,)
    accuracy = sum(Ys == Y_hats)/len(Ys)
    loss = loss/len(Ys)
    return loss, accuracy


if __name__ == "__main__":
    """The dataset loaders were provided for you.
    You need to implement your own training process.
    You need plot the loss and accuracies during the training process and test process. 
    """
    import pickle

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

    #model
    model = SingleLayerMLP(indim, outdim, hidden_dim).to(device).to(torch.double)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    losses, accuracies, losses2, accuracies2 = [], [], [], []
    for i in range(epochs):
        epoch = i + 1
        #train
        epoch_losses, epoch_accuracies, epoch_chunks = [], [], []
        for xs, ys in train_loader:
            xs = xs.to(device).to(torch.double)
            ys = ys.to(device).to(torch.long)
            output = model.forward(xs)
            loss = criterion.forward(output, ys)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss, epoch_accuracy = validate(train_loader)
        epoch_loss2, epoch_accuracy2 = validate(test_loader)
        print(f"Epoch: {epoch}  Train Loss: {epoch_loss}  Train Accuracy: {epoch_accuracy}", end="  ")
        print(f"Test Loss: {epoch_loss2}  Test Accuracy: {epoch_accuracy2}")
        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)
        losses2.append(epoch_loss2)
        accuracies2.append(epoch_accuracy2)

        if epoch_accuracy2 >= 1.0:
            break
    pickle.dump((losses, accuracies, losses2, accuracies2), open("metrics_ref.pkl", "wb"))
