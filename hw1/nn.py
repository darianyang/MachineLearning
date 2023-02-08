"""
You will need to implement a single layer neural network from scratch.
IMPORTANT:
    DO NOT change any function signatures
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Tuple, Dict

import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class Transform(object):
    """
    This is the base class. You do not need to change anything.
    Read the comments in this class carefully.
    """
    def __init__(self):
        """
        Initialize any parameters
        """
        pass

    def forward(self, x):
        """
        x should be passed as column vectors
        """
        pass

    def backward(self, grad_wrt_out):
        """
        Compute and save the gradients wrt the parameters for step()
        Return grad_wrt_x which will be the grad_wrt_out for previous Transform
        """
        pass

    def step(self):
        """
        Apply gradients to update the parameters
        """
        pass

    def zerograd(self):
        """
        This is used to Reset the gradients.
        Usually called before backward()
        """
        pass


class ReLU(Transform):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        """
        x shape (indim, batch_size)
        return shape (indim, batch_size)
        """
        # h = ReLU(q) = max(0,q)
        return x * (x > 0)


    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        """
        raise NotImplementedError()


class LinearMap(Transform):
    def __init__(self, indim, outdim, lr=0.001):
        """
        indim: input dimension
        outdim: output dimension
        lr: learning rate
        """
        super(LinearMap, self).__init__()
        self.weights = 0.01 *torch.rand((outdim, indim), dtype=torch.float64, requires_grad=True, device=device)
        self.bias = 0.01 * torch.rand((outdim, 1), dtype=torch.float64, requires_grad=True, device=device)
        self.lr = lr


    def forward(self, x):
        """
        x shape (indim, batch_size)
        return shape (outdim, batch_size)
        """
        # q = W * x + B
        return (self.weights * x) + self.bias


    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        return shape (indim, batch_size)
        """
        #compute grad_wrt_weights
        
        #compute grad_wrt_bias
        
        #compute & return grad_wrt_input
        
        raise NotImplementedError()


    def step(self):
        """
        apply gradients calculated by backward() to update the parameters
        """
        raise NotImplementedError()


class SoftmaxCrossEntropyLoss(object):
    def forward(self, logits, labels):
        """
        logits are pre-softmax scores, labels are one-hot labels of given inputs
        logits and labels are in the shape of (num_classes, batch_size)
        returns loss as a scalar (i.e. mean value of the batch_size loss)
        """
        raise NotImplementedError()


    def backward(self):
        """
        return grad_wrt_logits shape (num_classes, batch_size)
        (don't forget to divide by batch_size because your loss is a mean)
        """
        raise NotImplementedError()
    
    def getAccu(self):
        """
        return accuracy here
        """
        raise NotImplementedError()


class SingleLayerMLP(Transform):
    """
    constructing a single layer neural network with the previous functions
    """
    def __init__(self, indim, outdim, hidden_layer=100, lr=0.001):
        super(SingleLayerMLP, self).__init__()
        raise NotImplementedError()


    def forward(self, x):
        """
        x shape (indim, batch_size)
        return the presoftmax logits shape(outdim, batch_size)
        """
        raise NotImplementedError()


    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        calculate the gradients wrt the parameters
        """
        raise NotImplementedError()

    
    def step(self):
        """update model parameters"""
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

def labels2onehot(labels: np.ndarray):
    return np.array([[i==lab for i in range(2)] for lab in labels]).astype(int)

if __name__ == "__main__":
    """
    The dataset loaders were provided for you.
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
    Xtrain = np.loadtxt("data/XTrain.txt", delimiter="\t")
    Ytrain = np.loadtxt("data/yTrain.txt", delimiter="\t").astype(int)
    m1, n1 = Xtrain.shape
    print("Xtrain shape:", m1, n1)
    train_ds = DS(Xtrain, Ytrain)
    train_loader = DataLoader(train_ds, batch_size=batch_size)

    # data exploration
    train_features, train_labels = next(iter(train_loader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    #print(train_features)

    Xtest = np.loadtxt("data/XTest.txt", delimiter="\t")
    Ytest = np.loadtxt("data/yTest.txt", delimiter="\t").astype(int)
    m2, n2 = Xtest.shape
    print("Xtest shape:", m1, n2)
    test_ds = DS(Xtest, Ytest)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # construct the model then
    # construct the training process
    #raise NotImplementedError()


    # what are the steps after loading data?
    
    # then non-linearity through ReLU
    # feed output of ReLU to next linear transformation
    # apply softmax to transform to probabilities
    # then calc cross-entropy from probabilities
    # classification is difference of labels
    # probabilty of labels gets accuracy
    # I will define forward and backwards functions for each step

    # TODO: prob need to adjust this data sizes and inputs eventually
    # for now, this is a good test system
    # first linear mapping 
    lm = LinearMap(indim, batch_size)
    lm_out = lm.forward(train_features)
    # then activation function transform
    relu_out = ReLU().forward(lm_out)
    # then another linear mapping
    lm2_out = LinearMap(indim, batch_size).forward(train_features)
    # then softmax probability transform
    # then calculate loss (cross-entropy)

    # plot to check
    plt.plot(lm2_out.detach().numpy())
    plt.show()


