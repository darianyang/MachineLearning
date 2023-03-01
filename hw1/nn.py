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
        self.x = x
        self.y = x * (x > 0)
        return self.y

    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        """
        #compute & return grad_wrt_input (dE/dX) 
        # TODO: which to calc grad for?
        # get deep copy of dE/dY
        self.x_grad = grad_wrt_out.clone()
        # derivative of ReLU
        self.x_grad[grad_wrt_out < 0] = 0
        self.x_grad[grad_wrt_out > 0] = 1
        return self.x_grad


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
        return y shape (outdim, batch_size)
        """
        # q = W * x + B
        self.x = x
        self.y = (self.weights * x) + self.bias
        return self.y

    def backward(self, grad_wrt_out):
        """
        dE/dY = grad_wrt_out : shape (outdim, batch_size)
        return shape (indim, batch_size)
        """
        #compute grad_wrt_weights (dE/dW = X.T * dE/dY)
        self.w_grad = self.x.t() * grad_wrt_out
        
        #compute grad_wrt_bias (dE/dB = dE/dY)
        self.b_grad = grad_wrt_out

        #compute & return grad_wrt_input (dE/dX = dE/dY * W.T)
        self.x_grad = grad_wrt_out * self.weights.t()
        
        return self.x_grad

    def step(self):
        """
        apply gradients calculated by backward() to update the parameters
        """
        self.weights -= self.lr * self.w_grad
        self.bias -= self.lr * self.b_grad


class SoftmaxCrossEntropyLoss(object):
    def stable_softmax(self, X):
        exps = torch.exp(X - torch.max(X))
        return exps / torch.sum(exps)

    def forward(self, logits, labels):
        """
        logits are pre-softmax scores, labels are one-hot labels of given inputs
        logits and labels are in the shape of (num_classes, batch_size) (actually reversed (TODO))
        returns loss as a scalar (i.e. mean value of the batch_size loss)
        """
        # label shape for number of classes
        self.n_classes = labels.shape[1]
        # calc softmax of logits
        self.logits = self.stable_softmax(logits)
        # get deep copy of label tensor ?
        #self.labels = labels.clone()
        self.labels = labels # TODO
        # convert labels to max vals 
        #self.labels = labels.argmax(axis=1)
        #print(self.labels)
        # return softmax cross entropy
        self.log_likelihood = -torch.log(self.logits[range(self.n_classes),self.labels])
        self.loss = torch.sum(self.log_likelihood) / self.n_classes
        return self.loss

    def backward(self):
        """
        return grad_wrt_logits shape (num_classes, batch_size)
        (don't forget to divide by batch_size because your loss is a mean)
        """
        # calc grad of the output
        #grad = self.stable_softmax(self.log_likelihood)
        grad = self.logits
        grad[range(self.n_classes),self.labels] -= 1
        grad /= self.n_classes
        return grad
    
    def getAccu(self):
        """
        return accuracy here
        """
        # accuracy is percent of correct labels
        # so comparing softmax(logits) probabilities to labels
        # like binary mapping? 
        # TODO
        print("ARGMAX: ", torch.argmax(self.logits))
        print(self.log_likelihood[0])
        print(self.labels)
        self.acc = np.sum(self.logits == self.labels)
        return self.acc


class SingleLayerMLP(Transform):
    """
    constructing a single layer neural network with the previous functions
    """
    def __init__(self, indim, outdim, hidden_layer=100, lr=0.001):
        super(SingleLayerMLP, self).__init__()
        self.lm1 = LinearMap(self.indim, self.outdim, self.lr)
        self.relu = ReLU()
        self.lm2 = LinearMap(self.indim, self.outdim, self.lr)

    def forward(self, x):
        """
        x shape (indim, batch_size)
        return the presoftmax logits shape(outdim, batch_size)
        """
        # first linear transform
        self.y = self.lm1.forward(x)
        # non-linearity with ReLU activation function
        self.y = self.relu.forward(self.y)
        # second linear transform --> logits (map to 0-1 probabilities)
        self.y = self.lm2.forward(self.y)
        # calc loss via cross-entropy? TODO
        return self.y

    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        calculate the gradients wrt the parameters
        """
        # grad from loss via cross-entropy? TODO

        # grad from second linear transform --> logits (map to 0-1 probabilities)
        self.x = self.lm2.backward(grad_wrt_out)
        # grad from non-linearity with ReLU activation function
        self.x = self.relu.backward(self.x)
        # grad from first linear transform
        self.x = self.lm1.backward(self.x)
        return self.x
    
    def step(self):
        """update model parameters"""
        self.lm1.step()
        self.lm2.step()


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


    # TODO: prob need to adjust this data sizes and inputs eventually
    # for now, this is a good test system
    # first linear mapping 
    lm = LinearMap(indim, batch_size)
    lm_out = lm.forward(train_features)
    # then activation function transform
    relu_out = ReLU().forward(lm_out)
    # then another linear mapping
    lm2_out = LinearMap(indim, batch_size).forward(train_features)
    
    # testing torch cross-entropy function first
    # import torch.nn
    # loss = torch.nn.CrossEntropyLoss()
    # test_features, test_labels = next(iter(test_loader))
    # loss_out = loss(lm2_out, test_features)
    # print(loss.forward(lm2_out, test_features))

    #print(lm2_out[0])
    #print(labels2onehot(train_labels).shape)
    
    # calc softmax probability transform and calculate loss (cross-entropy)
    ce_out = SoftmaxCrossEntropyLoss()
    loss = ce_out.forward(lm2_out, labels2onehot(train_labels))
    print(loss)
    acc = ce_out.getAccu()
    print(acc)
    # TODO: currently need to fix accuracy calc with binary classes
    # I should be getting probabilities for each class (1 or 0) and thus get argmax
    # of each vector which will be the prediction, then can compare how many of my y_pred
    # match with y_true labels

    # plot to check
    # plt.plot(ce_out.logits.detach().numpy())
    # plt.show()

    
    # apply softmax to transform to probabilities
    # then calc cross-entropy from probabilities

