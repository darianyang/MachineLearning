"""
CNN for image classification (HW2).
"""

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# TODO: put the png reader in get_item for dataset class
# read a png image
f_path = "../data/0961638616-nndl-for-science-hw2"
img = torchvision.io.read_image(f"{f_path}/train/C4thin_original_IMG_20150608_170038_cell_96.png")

# display the properties of image
print("Image data:", img)
print(img.size())
print(type(img))

# display the png image
# convert the image tensor to PIL image
img = torchvision.transforms.ToPILImage()(img)

# display the PIL image
#img.show()

class DS(Dataset):
    """
    Data and labels.
    """
    def __init__(self, root_dir, labels):
        self.root_dir = root_dir
        self.labels = pd.read_csv(labels)
        self.length = len(self.labels)

    def __getitem__(self, idx):
        # convert to list if input index is tensor
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get image name based on index
        img_name = os.path.join()

        # image = 
        # label = self.labels[idx]

        sample = {'image':image, 'label':label}
        return sample

    def __len__(self):
        return self.length


class CNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# TODO load test and train datasets

net = CNNet()

# TODO: use softmax cross-entropy loss

# loss function and optimizer
import torch.optim as optim

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# for epoch in range(2):  # loop over the dataset multiple times

#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         # print statistics
#         running_loss += loss.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
#             running_loss = 0.0

# print('Finished Training')