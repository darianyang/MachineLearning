"""
You will need to implement a single layer neural network from scratch.
IMPORTANT:
    DO NOT change any function signatures
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Tuple, Dict


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
        self.input = x
        relu_mask = x > 0
        x2 = x*relu_mask
        return x2

    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        """
        positive = self.input > 0
        grad_wrt_input = grad_wrt_out*positive
        return grad_wrt_input


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
        m, n = x.shape
        self.input = x
        x2 = torch.matmul(self.weights, x) + torch.tile(self.bias, (1, n))
        return x2


    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        return shape (indim, batch_size)
        """
        #grad_wrt_weights
        self.grad_wrt_weights = torch.matmul(grad_wrt_out, self.input.T)
        #grad_wrt_bias
        self.grad_wrt_bias = torch.sum(grad_wrt_out, axis=1).reshape(-1, 1)
        #grad_wrt_input
        grad_wrt_input = torch.matmul(self.weights.T, grad_wrt_out)
        return grad_wrt_input


    def step(self):
        """
        apply gradients calculated by backward() to update the parameters
        """
        self.weights -= self.lr * self.grad_wrt_weights
        self.bias -= self.lr * self.grad_wrt_bias


class SoftmaxCrossEntropyLoss(object):
    def forward(self, logits, labels):
        """
        logits are pre-softmax scores, labels are one-hot labels of given inputs
        logits and labels are in the shape of (num_classes, batch_size)
        returns loss as a scalar (i.e. mean value of the batch_size loss)
        """
        m, n = logits.shape
        denominator = torch.tile(torch.sum(torch.exp(logits), axis=0).reshape(1, n), (m, 1))
        probs = torch.exp(logits)/denominator
        loss = torch.mean(-torch.sum(torch.log(probs)*labels, axis=0))

        self.labels = labels
        self.probs = probs
        self.batch_size = n
        return loss

    def backward(self):
        """
        return shape (num_classes, batch_size)
        (don't forget to divide by batch_size because your loss is a mean)
        """
        grads = (self.probs - self.labels) / self.batch_size
        return grads
    
    def getAccu(self):
        """
        return accuracy here
        """
        predictions = torch.argmax(self.probs, axis=0)
        m, n = self.labels.shape
        correct_counts = 0
        for i in range(n):
            correct_counts += self.labels[predictions[i], i]
        accuracy = correct_counts/n
        return accuracy


class SingleLayerMLP(Transform):
    def __init__(self, indim, outdim, hidden_layer=100, lr=0.001):
        super(SingleLayerMLP, self).__init__()
        self.linear1 = LinearMap(indim, hidden_layer, lr)
        self.relu = ReLU()
        self.linear2 = LinearMap(hidden_layer, outdim, lr)


    def forward(self, x):
        """
        x shape (indim, batch_size)
        """
        x1 = self.linear1.forward(x)
        x2 = self.relu.forward(x1)
        x3 = self.linear2.forward(x2)
        return x3


    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)"""
        grad_wrt_input2 = self.linear2.backward(grad_wrt_out)
        grad_wrt_output1 = self.relu.backward(grad_wrt_input2)
        grad_wrt_input1 = self.linear1.backward(grad_wrt_output1)

    
    def step(self):
        self.linear1.step()
        self.linear2.step()


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
    model = SingleLayerMLP(indim, outdim, hidden_dim, lr)
    criterion = SoftmaxCrossEntropyLoss()
    losses, accuracies, losses2, accuracies2 = [], [], [], []
    for i in range(epochs):
        epoch = i + 1
        #train
        epoch_losses, epoch_accuracies, epoch_chunks = [], [], []
        for xs, ys in train_loader:
            xs = torch.tensor(xs.T, device=device)
            one_hot_labels = labels2onehot(ys)
            ys = torch.tensor(one_hot_labels.T, device=device)

            model.zerograd()
            output = model.forward(xs)
            loss = criterion.forward(output, ys)
            grads_wrt_logits = criterion.backward()
            model.backward(grads_wrt_logits)
            model.step()
            accuracy = criterion.getAccu()

            epoch_losses.append(loss.item())
            epoch_accuracies.append(accuracy.item())
            epoch_chunks.append(xs.shape[1])
        epoch_loss = np.dot(epoch_losses, epoch_chunks)/m1
        epoch_accuracy = np.dot(epoch_accuracies, epoch_chunks)/m1


        #test
        epoch_losses2, epoch_accuracies2, epoch_chunks2 = [], [], []
        for xs, ys in test_loader:
            xs = torch.tensor(xs.T, device=device)
            one_hot_labels = labels2onehot(ys)
            ys = torch.tensor(one_hot_labels.T, device=device)

            # model.zerograd()
            output = model.forward(xs)
            loss = criterion.forward(output, ys)
            # grads_wrt_logits = criterion.backward()
            # model.backward(grads_wrt_logits)
            # model.step()
            accuracy = criterion.getAccu()

            epoch_losses2.append(loss.item())
            epoch_accuracies2.append(accuracy.item())
            epoch_chunks2.append(xs.shape[1])
        epoch_loss2 = np.dot(epoch_losses2, epoch_chunks2)/m2
        epoch_accuracy2 = np.dot(epoch_accuracies2, epoch_chunks2)/m2
        
        print(f"Epoch: {epoch}  Train Loss: {epoch_loss}  Train Accuracy: {epoch_accuracy}", end="  ")
        print(f"Test Loss: {epoch_loss2}  Test Accuracy: {epoch_accuracy2}")
        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)
        losses2.append(epoch_loss2)
        accuracies2.append(epoch_accuracy2)

        if epoch_accuracy2 >= 1.0:
            break
    pickle.dump((losses, accuracies, losses2, accuracies2), open("metrics.pkl", "wb"))
